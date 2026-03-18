#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include "raylib/src/raylib.h"

#include "dsp/optimizer.h"
#include "dsp/filter.h"



constexpr Color argb(unsigned int col)
{
	return {
		(unsigned char)((col >> 16) & 0xFF),
		(unsigned char)((col >> 8) & 0xFF),
		(unsigned char)((col >> 0) & 0xFF),
		(unsigned char)((col >> 24) & 0xFF)
	};
}

static float LogMapFreqToX(float freq, float fmin, float fmax, float x0, float x1)
{
	const float lf = std::log(std::max(freq, 1.0f));
	const float l0 = std::log(fmin);
	const float l1 = std::log(fmax);
	const float t = (lf - l0) / (l1 - l0);
	return x0 + (x1 - x0) * t;
}

static float MapDBToY(float db, float dbMin, float dbMax, float y0, float y1)
{
	const float t = (db - dbMin) / (dbMax - dbMin);
	return y1 + (y0 - y1) * t;
}

static void DrawResponseCurve(const std::vector<float>& magDB,
	const MatchedIIRDesign& design,
	float x0, float y0, float x1, float y1,
	float dbMin, float dbMax,
	Color col)
{
	const int n = design.GetNumPoints();
	for (int i = 0; i < n - 1; ++i)
	{
		const float f0 = design.GetFreqAt(i);
		const float f1 = design.GetFreqAt(i + 1);

		const float px0 = LogMapFreqToX(f0, 20.0f, 24000.0f, x0, x1);
		const float px1 = LogMapFreqToX(f1, 20.0f, 24000.0f, x0, x1);

		const float py0 = MapDBToY(magDB[i], dbMin, dbMax, y0, y1);
		const float py1 = MapDBToY(magDB[i + 1], dbMin, dbMax, y0, y1);

		DrawLineEx({ px0, py0 }, { px1, py1 }, 2.0f, col);
	}
}

static void DrawResponseCurveDashed(const float* magDB,
	const MatchedIIRDesign& design,
	float x0, float y0, float x1, float y1,
	float dbMin, float dbMax,
	Color col,
	int dashLen = 6,
	int gapLen = 4)
{
	const int n = design.GetNumPoints();
	int patternCounter = 0;

	for (int i = 0; i < n - 1; ++i)
	{
		const float f0 = design.GetFreqAt(i);
		const float f1 = design.GetFreqAt(i + 1);

		const float px0 = LogMapFreqToX(f0, 20.0f, 24000.0f, x0, x1);
		const float px1 = LogMapFreqToX(f1, 20.0f, 24000.0f, x0, x1);

		const float py0 = MapDBToY(magDB[i], dbMin, dbMax, y0, y1);
		const float py1 = MapDBToY(magDB[i + 1], dbMin, dbMax, y0, y1);

		if (patternCounter < dashLen)
			DrawLineEx({ px0, py0 }, { px1, py1 }, 2.0f, col);

		++patternCounter;
		if (patternCounter >= dashLen + gapLen)
			patternCounter = 0;
	}
}
static void DrawAnyCurve(const std::vector<float>& vals,
	const MatchedIIRDesign& design,
	float x0, float y0, float x1, float y1,
	float vMin, float vMax,
	Color col,
	float thickness = 2.0f)
{
	const int n = design.GetNumPoints();
	for (int i = 0; i < n - 1; ++i)
	{
		const float f0 = design.GetFreqAt(i);
		const float f1 = design.GetFreqAt(i + 1);

		const float px0 = LogMapFreqToX(f0, 20.0f, 24000.0f, x0, x1);
		const float px1 = LogMapFreqToX(f1, 20.0f, 24000.0f, x0, x1);

		const float py0 = MapDBToY(vals[i], vMin, vMax, y0, y1);
		const float py1 = MapDBToY(vals[i + 1], vMin, vMax, y0, y1);

		DrawLineEx({ px0, py0 }, { px1, py1 }, thickness, col);
	}
}
int main1()
{
	InitWindow(1280, 800, "MatchedFilterDesign");
	SetTargetFPS(60);

	MatchedIIRDesign design;
	design.SetupAnalogPrototype(AnalogFilterType::LP, 12000.0f, 15.07f, 15.0f, 1.0f);

	std::vector<std::vector<float>> history;
	int totalIterations = 0;

	const float left = 80.0f;
	const float top = 40.0f;
	const float right = 1240.0f;
	const float bottom = 740.0f;

	const float dbMin = -80.0f;
	const float dbMax = 40.0f;

	const bool drawErrorCurve = true;
	const int errMode = 0; // 0 = dB error, 1 = linear ratio error in dB
	const float errMin = -24.0f;
	const float errMax = 24.0f;

	while (!WindowShouldClose())
	{
		int iterPerTime = 1;
		design.RunOptimizer(iterPerTime);
		//design.RunOptimizerDirect();
		totalIterations += iterPerTime;

		std::vector<float> nowResp;
		std::vector<float> errCurve;
		design.GetResponseDB(nowResp);
		design.GetErrorCurveDB(errCurve, errMode);
		history.push_back(nowResp);

		if (history.size() > 240)
			history.erase(history.begin());

		BeginDrawing();
		ClearBackground(argb(0xff000000));

		DrawRectangleLinesEx({ left, top, right - left, bottom - top }, 1.0f, argb(0xff404040));

		for (int d = -80; d <= 40; d += 10)
		{
			const float y = MapDBToY((float)d, dbMin, dbMax, top, bottom);
			DrawLine((int)left, (int)y, (int)right, (int)y, argb(0xff202020));
			DrawText(TextFormat("%d dB", d), 10, (int)y - 10, 16, argb(0xff808080));
		}

		const float freqMarks[] = { 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 };
		for (float f : freqMarks)
		{
			const float x = LogMapFreqToX(f, 20.0f, 24000.0f, left, right);
			DrawLine((int)x, (int)top, (int)x, (int)bottom, argb(0xff202020));
			DrawText(TextFormat("%g", f), (int)x - 16, (int)bottom + 8, 16, argb(0xff808080));
		}

		const int histCount = (int)history.size();
		for (int i = 0; i < histCount; ++i)
		{
			const float t = histCount > 1 ? (float)i / (float)(histCount - 1) : 0.0f;
			Color c = ColorFromHSV(270.0f * (1.0f - t), 1.0f, 1.0f);
			DrawResponseCurve(history[i], design, left, top, right, bottom, dbMin, dbMax, c);
		}

		DrawResponseCurveDashed(
			design.GetMagDBSpace(),
			design,
			left, top, right, bottom,
			dbMin, dbMax,
			RAYWHITE,
			6, 0
		);
		if (drawErrorCurve)
		{
			DrawAnyCurve(
				errCurve,
				design,
				left, top, right, bottom,
				errMin, errMax,
				argb(0xff777777),
				2.0f
			);
		}

		DrawText("Prototype: 2.5-order Lowpass, fc=1000 Hz, Q=10", 20, 10, 20, RAYWHITE);
		DrawText(TextFormat("Iterations: %d", totalIterations), 900, 10, 20, RAYWHITE);

		EndDrawing();
	}

	CloseWindow();
	return 0;
}


static float Clamp01(float x)
{
	if (x < 0.0f) return 0.0f;
	if (x > 1.0f) return 1.0f;
	return x;
}

static float MapLinear(float v, float v0, float v1, float x0, float x1)
{
	const float t = (v - v0) / (v1 - v0);
	return x0 + (x1 - x0) * t;
}

static float ComputeRMSE(const std::vector<float>& v)
{
	if (v.empty()) return 0.0f;
	double s = 0.0;
	for (float x : v) s += (double)x * (double)x;
	return (float)std::sqrt(s / (double)v.size());
}

static float ComputeMaxAbs(const std::vector<float>& v)
{
	float m = 0.0f;
	for (float x : v)
	{
		const float a = fabsf(x);
		if (a > m) m = a;
	}
	return m;
}

static void DrawLinearCurve(const std::vector<float>& vals,
	float x0, float y0, float x1, float y1,
	float vMin, float vMax,
	Color col,
	float thickness = 2.0f)
{
	const int n = (int)vals.size();
	if (n < 2) return;

	for (int i = 0; i < n - 1; ++i)
	{
		const float px0 = MapLinear((float)i, 0.0f, (float)(n - 1), x0, x1);
		const float px1 = MapLinear((float)(i + 1), 0.0f, (float)(n - 1), x0, x1);

		const float py0 = MapDBToY(vals[i], vMin, vMax, y0, y1);
		const float py1 = MapDBToY(vals[i + 1], vMin, vMax, y0, y1);

		DrawLineEx({ px0, py0 }, { px1, py1 }, thickness, col);
	}
}

static void DrawPanelBox(float x0, float y0, float x1, float y1, const char* title)
{
	DrawRectangleRec({ x0, y0, x1 - x0, y1 - y0 }, argb(0xff111111));
	DrawRectangleLinesEx({ x0, y0, x1 - x0, y1 - y0 }, 1.0f, argb(0xff404040));
	DrawText(title, (int)x0 + 8, (int)y0 + 6, 20, RAYWHITE);
}

static void DrawResponseGrid(float x0, float y0, float x1, float y1, float dbMin, float dbMax)
{
	for (int d = (int)dbMin; d <= (int)dbMax; d += 10)
	{
		const float y = MapDBToY((float)d, dbMin, dbMax, y0, y1);
		DrawLine((int)x0, (int)y, (int)x1, (int)y, argb(0xff202020));
		DrawText(TextFormat("%d", d), (int)x0 + 4, (int)y - 8, 14, argb(0xff808080));
	}

	const float freqMarks[] = { 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 };
	for (float f : freqMarks)
	{
		const float x = LogMapFreqToX(f, 20.0f, 24000.0f, x0, x1);
		DrawLine((int)x, (int)y0, (int)x, (int)y1, argb(0xff202020));
		DrawText(TextFormat("%g", f), (int)x - 16, (int)y1 - 18, 14, argb(0xff808080));
	}
}

static void DrawConvergenceGrid(float x0, float y0, float x1, float y1, int maxIter, float vMin, float vMax)
{
	for (int i = 0; i <= 8; ++i)
	{
		const float xv = (float)i / 8.0f * (float)maxIter;
		const float x = MapLinear(xv, 0.0f, (float)maxIter, x0, x1);
		DrawLine((int)x, (int)y0, (int)x, (int)y1, argb(0xff202020));
		DrawText(TextFormat("%d", (int)xv), (int)x - 10, (int)y1 - 18, 14, argb(0xff808080));
	}
	for (int d = (int)vMin; d <= (int)vMax; d += 2)
	{
		const float y = MapDBToY((float)d, vMin, vMax, y0, y1);
		DrawLine((int)x0, (int)y, (int)x1, (int)y, argb(0xff202020));
		DrawText(TextFormat("%d", d), (int)x0 + 4, (int)y - 8, 14, argb(0xff808080));
	}
}

static void DrawConvergenceCurve(const std::vector<float>& vals,
	float x0, float y0, float x1, float y1,
	int maxIter,
	float vMin, float vMax,
	Color col,
	float thickness = 2.0f)
{
	const int n = (int)vals.size();
	if (n < 2) return;

	for (int i = 0; i < n - 1; ++i)
	{
		const float it0 = (float)i / (float)(n - 1) * (float)maxIter;
		const float it1 = (float)(i + 1) / (float)(n - 1) * (float)maxIter;

		const float px0 = MapLinear(it0, 0.0f, (float)maxIter, x0, x1);
		const float px1 = MapLinear(it1, 0.0f, (float)maxIter, x0, x1);

		const float py0 = MapDBToY(vals[i], vMin, vMax, y0, y1);
		const float py1 = MapDBToY(vals[i + 1], vMin, vMax, y0, y1);

		DrawLineEx({ px0, py0 }, { px1, py1 }, thickness, col);
	}
}

struct BenchResult
{
	const char* name = "";
	Color color = RAYWHITE;

	std::vector<float> finalRespDB;
	std::vector<float> finalErrDB;
	std::vector<float> convRMSE;
	std::vector<float> convMaxAbs;

	float finalRMSE = 0.0f;
	float finalMaxAbs = 0.0f;
};

int main2()
{
	InitWindow(1500, 920, "IIR Design Benchmark");
	SetTargetFPS(60);

	constexpr int numTypes = 4;
	const char* typeNames[numTypes] =
	{
		"FourStageNonlinearWhiteningIIR",
		"FourStageRealIIR",
		"TwoStageComplexIIR",
		"TwoStageCosIIR"
	};

	const Color typeColors[numTypes] =
	{
		argb(0xff66ccff),
		argb(0xffffaa33),
		argb(0xff66ff88),
		argb(0xffff6688)
	};

	// 这里先固定一个相对难一点的目标，便于看出差异
	const AnalogFilterType protoType = AnalogFilterType::LP;
	const float fc = 12000.0f;
	const float q = 10.07f;
	const float gain = 15.0f;
	const float stages = 3.0f;

	const int maxIter = 400;
	const int sampleStep = 4;
	const int numSamples = maxIter / sampleStep + 1;

	BenchResult results[numTypes];
	bool benchDone = false;

	auto RunBenchmark = [&]()
		{
			for (int t = 0; t < numTypes; ++t)
			{
				results[t] = {};
				results[t].name = typeNames[t];
				results[t].color = typeColors[t];

				MatchedIIRDesign design(t);
				design.SetupAnalogPrototype(protoType, fc, q, gain, stages);

				std::vector<float> err;
				for (int iter = 0; iter <= maxIter; iter += sampleStep)
				{
					if (iter > 0)
						design.RunOptimizer(sampleStep);

					design.GetErrorCurveDB(err, 0);
					results[t].convRMSE.push_back(ComputeRMSE(err));
					results[t].convMaxAbs.push_back(ComputeMaxAbs(err));
				}

				design.GetResponseDB(results[t].finalRespDB);
				design.GetErrorCurveDB(results[t].finalErrDB, 0);
				results[t].finalRMSE = ComputeRMSE(results[t].finalErrDB);
				results[t].finalMaxAbs = ComputeMaxAbs(results[t].finalErrDB);
			}
		};

	RunBenchmark();
	benchDone = true;

	const float W = 1500.0f;
	const float H = 920.0f;

	const float margin = 20.0f;
	const float gap = 16.0f;

	const float leftW = 940.0f;
	const float rightW = W - margin * 2.0f - gap - leftW;

	const float panel1X0 = margin;
	const float panel1Y0 = margin;
	const float panel1X1 = panel1X0 + leftW;
	const float panel1Y1 = 430.0f;

	const float panel2X0 = margin;
	const float panel2Y0 = panel1Y1 + gap;
	const float panel2X1 = panel1X1;
	const float panel2Y1 = H - margin;

	const float panel3X0 = panel1X1 + gap;
	const float panel3Y0 = margin;
	const float panel3X1 = W - margin;
	const float panel3Y1 = H - margin;

	const float respDbMin = -80.0f;
	const float respDbMax = 30.0f;

	float errDbMin = 0.0f;
	float errDbMax = 12.0f;
	{
		float m = 0.0f;
		for (int t = 0; t < numTypes; ++t)
		{
			for (float v : results[t].convRMSE) if (v > m) m = v;
			if (results[t].finalMaxAbs > m) m = results[t].finalMaxAbs;
		}
		errDbMax = std::max(6.0f, std::ceil(m + 1.0f));
	}

	while (!WindowShouldClose())
	{
		BeginDrawing();
		ClearBackground(argb(0xff000000));

		DrawPanelBox(panel1X0, panel1Y0, panel1X1, panel1Y1, "Final Magnitude Response");
		DrawPanelBox(panel2X0, panel2Y0, panel2X1, panel2Y1, "Convergence Speed (RMSE dB)");
		DrawPanelBox(panel3X0, panel3Y0, panel3X1, panel3Y1, "Final Metrics");

		// Panel 1: 最终响应
		{
			const float x0 = panel1X0 + 50.0f;
			const float y0 = panel1Y0 + 36.0f;
			const float x1 = panel1X1 - 20.0f;
			const float y1 = panel1Y1 - 24.0f;

			DrawResponseGrid(x0, y0, x1, y1, respDbMin, respDbMax);

			MatchedIIRDesign targetRef(0);
			targetRef.SetupAnalogPrototype(protoType, fc, q, gain, stages);
			DrawResponseCurveDashed(
				targetRef.GetMagDBSpace(),
				targetRef,
				x0, y0, x1, y1,
				respDbMin, respDbMax,
				RAYWHITE,
				6, 0
			);

			for (int t = 0; t < numTypes; ++t)
			{
				DrawResponseCurve(
					results[t].finalRespDB,
					targetRef,
					x0, y0, x1, y1,
					respDbMin, respDbMax,
					results[t].color
				);
			}
		}

		// Panel 2: 收敛曲线
		{
			const float x0 = panel2X0 + 50.0f;
			const float y0 = panel2Y0 + 36.0f;
			const float x1 = panel2X1 - 20.0f;
			const float y1 = panel2Y1 - 24.0f;

			DrawConvergenceGrid(x0, y0, x1, y1, maxIter, errDbMin, errDbMax);

			for (int t = 0; t < numTypes; ++t)
			{
				DrawConvergenceCurve(
					results[t].convRMSE,
					x0, y0, x1, y1,
					maxIter,
					errDbMin, errDbMax,
					results[t].color,
					2.5f
				);
			}
		}

		// Panel 3: 条形结果 + legend
		{
			const float x0 = panel3X0 + 18.0f;
			const float y0 = panel3Y0 + 40.0f;
			const float x1 = panel3X1 - 18.0f;
			const float y1 = panel3Y1 - 18.0f;

			DrawText(TextFormat("Target: LP  fc=%.1f  Q=%.2f  gain=%.1f  stages=%.1f", fc, q, gain, stages),
				(int)x0, (int)y0, 18, RAYWHITE);

			float legendY = y0 + 34.0f;
			for (int t = 0; t < numTypes; ++t)
			{
				DrawRectangle((int)x0, (int)legendY + t * 24, 16, 16, results[t].color);
				DrawText(results[t].name, (int)x0 + 24, (int)legendY - 1 + t * 24, 18, RAYWHITE);
			}

			const float tableY = legendY + numTypes * 24 + 18.0f;
			DrawText("Final RMSE (dB)", (int)x0, (int)tableY, 18, RAYWHITE);
			DrawText("Final MaxAbs (dB)", (int)(x0 + 190), (int)tableY, 18, RAYWHITE);

			for (int t = 0; t < numTypes; ++t)
			{
				const float yy = tableY + 28.0f + t * 24.0f;
				DrawText(TextFormat("%.4f", results[t].finalRMSE), (int)x0, (int)yy, 18, results[t].color);
				DrawText(TextFormat("%.4f", results[t].finalMaxAbs), (int)(x0 + 190), (int)yy, 18, results[t].color);
			}

			const float chartTop = tableY + 150.0f;
			const float chartBottom = y1 - 20.0f;
			const float chartMid = chartTop + (chartBottom - chartTop) * 0.5f - 20.0f;

			DrawText("RMSE", (int)x0, (int)chartTop - 24, 18, RAYWHITE);
			DrawText("MaxAbs", (int)x0, (int)chartMid - 24, 18, RAYWHITE);

			float maxRMSE = 1.0f;
			float maxMaxAbs = 1.0f;
			for (int t = 0; t < numTypes; ++t)
			{
				if (results[t].finalRMSE > maxRMSE) maxRMSE = results[t].finalRMSE;
				if (results[t].finalMaxAbs > maxMaxAbs) maxMaxAbs = results[t].finalMaxAbs;
			}
			maxRMSE *= 1.15f;
			maxMaxAbs *= 1.15f;

			const float barX0 = x0 + 10.0f;
			const float barX1 = x1 - 10.0f;
			const float barW = (barX1 - barX0) / (float)numTypes - 18.0f;

			for (int t = 0; t < numTypes; ++t)
			{
				const float bx = barX0 + t * ((barX1 - barX0) / (float)numTypes) + 8.0f;

				{
					const float h = (chartMid - chartTop - 30.0f) * (results[t].finalRMSE / maxRMSE);
					const float by = chartMid - h;
					DrawRectangle((int)bx, (int)by, (int)barW, (int)h, results[t].color);
					DrawText(TextFormat("%.3f", results[t].finalRMSE), (int)bx, (int)by - 18, 16, RAYWHITE);
				}

				{
					const float h = (chartBottom - chartMid - 30.0f) * (results[t].finalMaxAbs / maxMaxAbs);
					const float by = chartBottom - h;
					DrawRectangle((int)bx, (int)by, (int)barW, (int)h, results[t].color);
					DrawText(TextFormat("%.3f", results[t].finalMaxAbs), (int)bx, (int)by - 18, 16, RAYWHITE);
				}

				DrawText(TextFormat("T%d", t), (int)(bx + barW * 0.35f), (int)chartBottom + 4, 16, argb(0xffc0c0c0));
			}
		}

		DrawText("IIR Topology Benchmark", 20, 4, 20, RAYWHITE);
		DrawText("Dashed white = target response", 1220, 4, 18, argb(0xffc0c0c0));

		EndDrawing();
	}

	CloseWindow();
	return 0;
}


int main()
{
	//main1();
	main2();
}