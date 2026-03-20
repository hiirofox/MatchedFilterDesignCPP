#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <vector>
#include <cmath>
#include <algorithm>

#include "raylib/src/raylib.h"
#include "dsp/optimizer.h"
#include "dsp/filter.h"
#include "dsp/analygrad.h"

constexpr Color argb(unsigned int col)
{
	return {
		(unsigned char)((col >> 16) & 0xFF),
		(unsigned char)((col >> 8) & 0xFF),
		(unsigned char)((col >> 0) & 0xFF),
		(unsigned char)((col >> 24) & 0xFF)
	};
}

static Font gMonoFont = { 0 };
static bool gMonoFontLoaded = false;

static void LoadUIFont()
{
	const char* candidates[] =
	{
		"C:/Windows/Fonts/FSEX300.ttf",
		"C:/Windows/Fonts/fixedsys.ttf",
		"C:/Windows/Fonts/consola.ttf",
		"C:/Windows/Fonts/lucon.ttf"
	};

	for (const char* p : candidates)
	{
		if (FileExists(p))
		{
			gMonoFont = LoadFontEx(p, 16, 0, 0);
			if (gMonoFont.texture.id != 0)
			{
				gMonoFontLoaded = true;
				return;
			}
		}
	}
}

static void UnloadUIFont()
{
	if (gMonoFontLoaded)
	{
		UnloadFont(gMonoFont);
		gMonoFontLoaded = false;
	}
}

static void DrawTextUI(const char* text, float x, float y, float fontSize, Color color)
{
	if (gMonoFontLoaded)
		DrawTextEx(gMonoFont, text, { x, y }, fontSize, 0.0f, color);
	else
		DrawText(text, (int)x, (int)y, (int)fontSize, color);
}

static void DrawTextFmt(float x, float y, float fontSize, Color color, const char* fmt, ...)
{
	char buf[1024];
	va_list args;
	va_start(args, fmt);
	vsnprintf(buf, sizeof(buf), fmt, args);
	va_end(args);
	DrawTextUI(buf, x, y, fontSize, color);
}
static Vector2 MeasureTextUI(const char* text, float fontSize)
{
	if (gMonoFontLoaded)
		return MeasureTextEx(gMonoFont, text, fontSize, 0.0f);
	return { (float)MeasureText(text, (int)fontSize), fontSize };
}

static void DrawFramedChartTitle(const char* text, float x0, float y0, float x1, float y1, float fontSize = 18.0f)
{
	DrawRectangleLinesEx({ x0, y0, x1 - x0, y1 - y0 }, 1.0f, argb(0xff606060));

	const Vector2 sz = MeasureTextUI(text, fontSize);
	const float padX = 8.0f;
	const float padY = 2.0f;

	const float tx = 0.5f * (x0 + x1) - sz.x * 0.5f;
	const float ty = y0 - sz.y * 0.5f - padY;

	// 用背景色盖住边框中间一小段，让文字“压”在线上
	DrawRectangleRec(
		{
			tx - padX,
			ty,
			sz.x + padX * 2.0f,
			sz.y + padY * 2.0f
		},
		argb(0xff111111)
	);

	DrawTextUI(text, tx, ty - 1.0f, fontSize, RAYWHITE);
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

static float MapLinear(float v, float v0, float v1, float x0, float x1)
{
	const float t = (v - v0) / (v1 - v0);
	return x0 + (x1 - x0) * t;
}

static void DrawPanelBox(float x0, float y0, float x1, float y1, const char* title)
{
	DrawRectangleRec({ x0, y0, x1 - x0, y1 - y0 }, argb(0xff111111));
	DrawRectangleLinesEx({ x0, y0, x1 - x0, y1 - y0 }, 1.0f, argb(0xff404040));
	DrawTextUI(title, x0 + 8, y0 + 6, 20.0f, RAYWHITE);
}

static void DrawResponseGrid(float x0, float y0, float x1, float y1, float dbMin, float dbMax)
{
	for (int d = (int)dbMin; d <= (int)dbMax; d += 10)
	{
		const float y = MapDBToY((float)d, dbMin, dbMax, y0, y1);
		DrawLine((int)x0, (int)y, (int)x1, (int)y, argb(0xff202020));
		DrawTextFmt(x0 + 4, y - 8, 14.0f, argb(0xff808080), "%d", d);
	}

	const float freqMarks[] = { 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 };
	for (float f : freqMarks)
	{
		const float x = LogMapFreqToX(f, 20.0f, 24000.0f, x0, x1);
		DrawLine((int)x, (int)y0, (int)x, (int)y1, argb(0xff202020));
		DrawTextFmt(x - 18, y1 - 18, 14.0f, argb(0xff808080), "%g", f);
	}
}

static void DrawConvergenceGrid(float x0, float y0, float x1, float y1, int maxIter, float vMin, float vMax)
{
	for (int i = 0; i <= 8; ++i)
	{
		const float xv = (float)i / 8.0f * (float)maxIter;
		const float x = MapLinear(xv, 0.0f, (float)maxIter, x0, x1);
		DrawLine((int)x, (int)y0, (int)x, (int)y1, argb(0xff202020));
		DrawTextFmt(x - 12, y1 - 18, 14.0f, argb(0xff808080), "%d", (int)xv);
	}

	for (int d = (int)vMin; d <= (int)vMax; d += 2)
	{
		const float y = MapDBToY((float)d, vMin, vMax, y0, y1);
		DrawLine((int)x0, (int)y, (int)x1, (int)y, argb(0xff202020));
		DrawTextFmt(x0 + 4, y - 8, 14.0f, argb(0xff808080), "%d", d);
	}
}
static void DrawCenteredLabelBox(const char* text, float centerX, float y, float fontSize)
{
	const Vector2 sz = MeasureTextUI(text, fontSize);

	const float padX = 10.0f;
	const float padY = 4.0f;
	const float w = sz.x + padX * 2.0f;
	const float h = sz.y + padY * 2.0f;
	const float x = centerX - w * 0.5f;

	DrawRectangleRec({ x, y, w, h }, argb(0xcc101010));
	DrawRectangleLinesEx({ x, y, w, h }, 1.0f, argb(0xff606060));
	DrawTextUI(text, x + padX, y + padY - 1.0f, fontSize, RAYWHITE);
}

static void DrawConvergenceCurve(
	const std::vector<float>& vals,
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

static std::vector<float> GenerateLogFC(float fcMin, float fcMax, int densityPerDecade)
{
	std::vector<float> fcList;
	const float logMin = std::log10(fcMin);
	const float logMax = std::log10(fcMax);
	const float decades = logMax - logMin;
	const int totalPoints = std::max(2, (int)(decades * densityPerDecade));

	fcList.reserve(totalPoints);
	for (int i = 0; i < totalPoints; ++i)
	{
		const float t = (float)i / (float)(totalPoints - 1);
		const float logf = logMin + t * (logMax - logMin);
		fcList.push_back(std::pow(10.0f, logf));
	}
	return fcList;
}

static void DrawDesignCurve(
	IIRDesignBase* design,
	bool drawBest,
	bool drawPrototype,
	float x0, float y0, float x1, float y1,
	float dbMin, float dbMax,
	Color col,
	float thickness = 2.0f,
	int plotPoints = 768,
	float fMin = 20.0f,
	float fMax = 24000.0f)
{
	if (!design) return;

	for (int i = 0; i < plotPoints - 1; ++i)
	{
		const float t0 = (float)i / (float)(plotPoints - 1);
		const float t1 = (float)(i + 1) / (float)(plotPoints - 1);

		const float f0 = std::exp(std::log(fMin) + (std::log(fMax) - std::log(fMin)) * t0);
		const float f1 = std::exp(std::log(fMin) + (std::log(fMax) - std::log(fMin)) * t1);

		const float m0 = drawPrototype
			? design->GetPrototypeResp(f0)
			: (drawBest ? design->GetBestIIRResp(f0) : design->GetNowIIRResp(f0));

		const float m1 = drawPrototype
			? design->GetPrototypeResp(f1)
			: (drawBest ? design->GetBestIIRResp(f1) : design->GetNowIIRResp(f1));

		const float db0 = 20.0f * std::log10(std::max(m0, 1.0e-30f));
		const float db1 = 20.0f * std::log10(std::max(m1, 1.0e-30f));

		const float px0 = LogMapFreqToX(f0, fMin, fMax, x0, x1);
		const float px1 = LogMapFreqToX(f1, fMin, fMax, x0, x1);

		const float py0 = MapDBToY(db0, dbMin, dbMax, y0, y1);
		const float py1 = MapDBToY(db1, dbMin, dbMax, y0, y1);

		DrawLineEx({ px0, py0 }, { px1, py1 }, thickness, col);
	}
}

static void BuildErrorCurve(
	IIRDesignBase* design,
	bool useBest,
	std::vector<float>& outErrDB,
	int plotPoints = 512,
	float fMin = 20.0f,
	float fMax = 24000.0f)
{
	outErrDB.resize(plotPoints);
	for (int i = 0; i < plotPoints; ++i)
	{
		const float t = (float)i / (float)(plotPoints - 1);
		const float f = std::exp(std::log(fMin) + (std::log(fMax) - std::log(fMin)) * t);

		const float refMag = std::max(design->GetPrototypeResp(f), 1.0e-30f);
		const float fitMag = std::max(useBest ? design->GetBestIIRResp(f) : design->GetNowIIRResp(f), 1.0e-30f);

		outErrDB[i] = 20.0f * std::log10(fitMag) - 20.0f * std::log10(refMag);
	}
}

static void DrawErrorCurve(
	const std::vector<float>& errDB,
	float x0, float y0, float x1, float y1,
	float errMin, float errMax,
	Color col,
	float thickness = 2.0f)
{
	const int n = (int)errDB.size();
	if (n < 2) return;

	for (int i = 0; i < n - 1; ++i)
	{
		const float t0 = (float)i / (float)(n - 1);
		const float t1 = (float)(i + 1) / (float)(n - 1);

		const float f0 = std::exp(std::log(20.0f) + (std::log(24000.0f) - std::log(20.0f)) * t0);
		const float f1 = std::exp(std::log(20.0f) + (std::log(24000.0f) - std::log(20.0f)) * t1);

		const float px0 = LogMapFreqToX(f0, 20.0f, 24000.0f, x0, x1);
		const float px1 = LogMapFreqToX(f1, 20.0f, 24000.0f, x0, x1);

		const float py0 = MapDBToY(errDB[i], errMin, errMax, y0, y1);
		const float py1 = MapDBToY(errDB[i + 1], errMin, errMax, y0, y1);

		DrawLineEx({ px0, py0 }, { px1, py1 }, thickness, col);
	}
}


static int TestOptimizationHistory()
{
	InitWindow(1280, 800, "TestOptimizationHistory");
	LoadUIFont();
	SetTargetFPS(60);

	using IIRDesigner = AnalyticGradient::MatchedComplexIIRDesignAnalytic;
	IIRDesigner design(2);
	//design.SetWarpThreshold(15000.0f);
	const AnalogFilterType protoType = AnalogFilterType::LP;
	const float q = 20.07f;
	const float gain = 12.0f;
	const float stages = 1.0f;
	design.SetupAnalogPrototype(protoType, 2000, q, gain, stages);

	std::vector<std::vector<float>> historyCoeffs;
	int totalIterations = 0;

	const float left = 80.0f;
	const float top = 40.0f;
	const float right = 1240.0f;
	const float bottom = 740.0f;

	const float dbMin = -80.0f;
	const float dbMax = 40.0f;
	const float errMin = -24.0f;
	const float errMax = 24.0f;

	while (!WindowShouldClose())
	{
		design.RunOptimizer(1, 400);
		++totalIterations;

		std::vector<float> nowCoeffs;
		design.GetNowCoeffs(nowCoeffs);
		historyCoeffs.push_back(nowCoeffs);
		if (historyCoeffs.size() > 140)
			historyCoeffs.erase(historyCoeffs.begin());

		std::vector<float> errCurve;
		BuildErrorCurve(&design, false, errCurve, 512);

		BeginDrawing();
		ClearBackground(argb(0xff000000));

		DrawRectangleLinesEx({ left, top, right - left, bottom - top }, 1.0f, argb(0xff404040));
		DrawResponseGrid(left, top, right, bottom, dbMin, dbMax);

		for (int i = 0; i < (int)historyCoeffs.size(); ++i)
		{
			IIRDesigner temp = design;
			temp.GetNowCoeffs(nowCoeffs); // no-op; keeps compiler quiet if needed
			const float t = historyCoeffs.size() > 1 ? (float)i / (float)(historyCoeffs.size() - 1) : 0.0f;
			Color c = ColorFromHSV(270.0f * (1.0f - t), 1.0f, 1.0f);

			// 直接从存下来的系数画
			std::vector<float> coeffs = historyCoeffs[i];
			for (int k = 0; k < 511; ++k)
			{
				const float tt0 = (float)k / 511.0f;
				const float tt1 = (float)(k + 1) / 511.0f;
				const float f0 = std::exp(std::log(20.0f) + (std::log(24000.0f) - std::log(20.0f)) * tt0);
				const float f1 = std::exp(std::log(20.0f) + (std::log(24000.0f) - std::log(20.0f)) * tt1);

				// 用当前设计对象的 warped 逻辑，只借接口不借旧表
				IIRDesigner local = design;
				local.GetNowCoeffs(coeffs); // 占位，不使用
				// 这里直接复用 design 当前对象不太方便强塞任意 coeff，所以历史轨迹仍建议简化成只画当前与目标
				// 故不再画历史表，改成只画当前响应
			}
		}

		DrawDesignCurve(&design, false, true, left, top, right, bottom, dbMin, dbMax, argb(0xffb0b0b0), 1.5f, 768);
		DrawDesignCurve(&design, false, false, left, top, right, bottom, dbMin, dbMax, ColorFromHSV(120.0f, 1.0f, 1.0f), 2.0f, 768);
		DrawErrorCurve(errCurve, left, top, right, bottom, errMin, errMax, argb(0xff666666), 1.5f);

		DrawTextUI("Optimization history / current response", 20, 8, 20.0f, RAYWHITE);
		DrawTextFmt(20, 30, 18.0f, argb(0xffc0c0c0), "iter=%d", totalIterations);
		DrawTextUI("light gray = prototype, green = current response, dark gray = error", 700, 8, 16.0f, argb(0xffc0c0c0));

		EndDrawing();
	}

	UnloadUIFont();
	CloseWindow();
	return 0;
}

struct BenchResult
{
	const char* name = "";
	Color color = RAYWHITE;
	std::vector<float> convRMSE;
	std::vector<float> convMaxAbs;
	float finalRMSE = 0.0f;
	float finalMaxAbs = 0.0f;
	WarpedMatchedIIRDesign* design = nullptr;
};

static int TestTopologyBenchmark()
{
	InitWindow(1500, 920, "TestTopologyBenchmark");
	LoadUIFont();
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

	const AnalogFilterType protoType = AnalogFilterType::HS;
	const float fc = 12000.0f;
	const float q = 10.07f;
	const float gain = 15.0f;
	const float stages = 3.0f;

	const int maxIter = 200;
	const int sampleStep = 4;

	BenchResult results[numTypes];

	for (int t = 0; t < numTypes; ++t)
	{
		results[t].name = typeNames[t];
		results[t].color = typeColors[t];
		results[t].design = new WarpedMatchedIIRDesign(t);
		results[t].design->SetWarpThreshold(15000.0f);
		results[t].design->SetupAnalogPrototype(protoType, fc, q, gain, stages);

		for (int iter = 0; iter <= maxIter; iter += sampleStep)
		{
			if (iter > 0)
				results[t].design->RunOptimizer(sampleStep, maxIter);

			std::vector<float> errCurve;
			BuildErrorCurve(results[t].design, false, errCurve, 512);
			results[t].convRMSE.push_back(ComputeRMSE(errCurve));
			results[t].convMaxAbs.push_back(ComputeMaxAbs(errCurve));
		}

		std::vector<float> finalErr;
		BuildErrorCurve(results[t].design, true, finalErr, 768);
		results[t].finalRMSE = ComputeRMSE(finalErr);
		results[t].finalMaxAbs = ComputeMaxAbs(finalErr);
	}

	const float W = 1500.0f;
	const float H = 920.0f;
	const float margin = 20.0f;
	const float gap = 16.0f;

	const float leftW = 940.0f;
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
	for (int t = 0; t < numTypes; ++t)
	{
		for (float v : results[t].convRMSE) errDbMax = std::max(errDbMax, v);
		errDbMax = std::max(errDbMax, results[t].finalMaxAbs);
	}
	errDbMax = std::max(6.0f, std::ceil(errDbMax + 1.0f));

	while (!WindowShouldClose())
	{
		BeginDrawing();
		ClearBackground(argb(0xff000000));

		DrawPanelBox(panel1X0, panel1Y0, panel1X1, panel1Y1, "Final Magnitude Response");
		DrawPanelBox(panel2X0, panel2Y0, panel2X1, panel2Y1, "Convergence Speed (RMSE dB)");
		DrawPanelBox(panel3X0, panel3Y0, panel3X1, panel3Y1, "Final Metrics");

		{
			const float x0 = panel1X0 + 50.0f;
			const float y0 = panel1Y0 + 36.0f;
			const float x1 = panel1X1 - 20.0f;
			const float y1 = panel1Y1 - 24.0f;

			DrawResponseGrid(x0, y0, x1, y1, respDbMin, respDbMax);

			DrawDesignCurve(results[0].design, true, true, x0, y0, x1, y1, respDbMin, respDbMax, RAYWHITE, 1.4f, 768);

			for (int t = 0; t < numTypes; ++t)
			{
				DrawDesignCurve(results[t].design, true, false, x0, y0, x1, y1, respDbMin, respDbMax, results[t].color, 2.2f, 768);
			}
		}

		{
			const float x0 = panel2X0 + 50.0f;
			const float y0 = panel2Y0 + 36.0f;
			const float x1 = panel2X1 - 20.0f;
			const float y1 = panel2Y1 - 24.0f;

			DrawConvergenceGrid(x0, y0, x1, y1, maxIter, errDbMin, errDbMax);

			for (int t = 0; t < numTypes; ++t)
			{
				DrawConvergenceCurve(results[t].convRMSE, x0, y0, x1, y1, maxIter, errDbMin, errDbMax, results[t].color, 2.5f);
			}
		}

		{
			const float x0 = panel3X0 + 18.0f;
			const float y0 = panel3Y0 + 40.0f;
			const float x1 = panel3X1 - 18.0f;
			const float y1 = panel3Y1 - 18.0f;

			DrawTextFmt(x0, y0, 18.0f, RAYWHITE,
				"Target: HS  fc=%.1f  Q=%.2f  gain=%.1f  stages=%.1f",
				fc, q, gain, stages);

			float legendY = y0 + 36.0f;
			for (int t = 0; t < numTypes; ++t)
			{
				DrawRectangle((int)x0, (int)legendY + t * 24, 16, 16, results[t].color);
				DrawTextUI(results[t].name, x0 + 26, legendY - 2 + t * 24, 17.0f, RAYWHITE);
			}

			const float tableY = legendY + numTypes * 24 + 20.0f;
			DrawTextUI("Final RMSE (dB)", x0, tableY, 18.0f, RAYWHITE);
			DrawTextUI("Final MaxAbs (dB)", x0 + 210, tableY, 18.0f, RAYWHITE);

			for (int t = 0; t < numTypes; ++t)
			{
				const float yy = tableY + 28.0f + t * 24.0f;
				DrawTextFmt(x0, yy, 18.0f, results[t].color, "%.4f", results[t].finalRMSE);
				DrawTextFmt(x0 + 210, yy, 18.0f, results[t].color, "%.4f", results[t].finalMaxAbs);
			}

			const float chartTop = tableY + 170.0f;
			const float chartBottom = y1 - 20.0f;
			const float chartMid = chartTop + (chartBottom - chartTop) * 0.5f - 24.0f;

			const float rmseBoxX0 = x0 + 6.0f;
			const float rmseBoxX1 = x1 - 6.0f;
			const float rmseBoxY0 = chartTop;
			const float rmseBoxY1 = chartMid - 16.0f;

			const float maxBoxX0 = x0 + 6.0f;
			const float maxBoxX1 = x1 - 6.0f;
			const float maxBoxY0 = chartMid + 18.0f;
			const float maxBoxY1 = chartBottom;

			// 整体框 + 标题压在线中间
			DrawFramedChartTitle("RMSE", rmseBoxX0, rmseBoxY0, rmseBoxX1, rmseBoxY1, 18.0f);
			DrawFramedChartTitle("MaxAbs", maxBoxX0, maxBoxY0, maxBoxX1, maxBoxY1, 18.0f);

			float maxRMSE = 1.0f;
			float maxMaxAbs = 1.0f;
			for (int t = 0; t < numTypes; ++t)
			{
				maxRMSE = std::max(maxRMSE, results[t].finalRMSE);
				maxMaxAbs = std::max(maxMaxAbs, results[t].finalMaxAbs);
			}
			maxRMSE *= 1.15f;
			maxMaxAbs *= 1.15f;

			const float barX0 = x0 + 14.0f;
			const float barX1 = x1 - 14.0f;
			const float slotW = (barX1 - barX0) / (float)numTypes;
			const float barW = slotW - 18.0f;

			// 两个图表框内部可用区域
			const float rmseInnerTop = rmseBoxY0 + 18.0f;
			const float rmseInnerBottom = rmseBoxY1 - 12.0f;
			const float maxInnerTop = maxBoxY0 + 18.0f;
			const float maxInnerBottom = maxBoxY1 - 12.0f;

			const float rmseAvailH = rmseInnerBottom - rmseInnerTop;
			const float maxAvailH = maxInnerBottom - maxInnerTop;

			for (int t = 0; t < numTypes; ++t)
			{
				const float bx = barX0 + t * slotW + 8.0f;

				{
					const float h = rmseAvailH * (results[t].finalRMSE / maxRMSE);
					const float by = rmseInnerBottom - h;
					DrawRectangle((int)bx, (int)by, (int)barW, (int)h, results[t].color);

					const float labelY = std::max(rmseInnerTop + 2.0f, by - 18.0f);
					DrawTextFmt(bx, labelY, 15.0f, RAYWHITE, "%.3f", results[t].finalRMSE);
				}

				{
					const float h = maxAvailH * (results[t].finalMaxAbs / maxMaxAbs);
					const float by = maxInnerBottom - h;
					DrawRectangle((int)bx, (int)by, (int)barW, (int)h, results[t].color);

					const float labelY = std::max(maxInnerTop + 2.0f, by - 18.0f);
					DrawTextFmt(bx, labelY, 15.0f, RAYWHITE, "%.3f", results[t].finalMaxAbs);
				}

				DrawTextFmt(bx + barW * 0.30f, maxBoxY1 + 4.0f, 15.0f, argb(0xffc0c0c0), "T%d", t);
			}
		}

		DrawTextUI("IIR topology benchmark", 20, 4, 20.0f, RAYWHITE);
		DrawTextUI("white = target response", 1260, 4, 17.0f, argb(0xffc0c0c0));

		EndDrawing();
	}

	for (int t = 0; t < numTypes; ++t)
		delete results[t].design;

	UnloadUIFont();
	CloseWindow();
	return 0;
}

static int TestFcSweepByParameterSpace()
{
	InitWindow(1600, 980, "TestFcSweepByParameterSpace");
	LoadUIFont();
	SetTargetFPS(5);

	constexpr int numTypes = 4;
	const char* typeNames[numTypes] =
	{
		"FourStageNonlinearWhiteningIIR",
		"FourStageRealIIR",
		"TwoStageComplexIIR",
		"TwoStageCosIIR"
	};

	const AnalogFilterType protoType = AnalogFilterType::LP;
	const float q = 12.07f;
	const float gain = 6.0f;
	const float stages = 1.0f;

	const float fcMin = 20.0f;
	const float fcMax = 36000.0f;
	const int densityPerDecade = 15;

	const float warpThresholdHz = 17000.0f;
	const int adamCycles = 20;
	const int lbfgsCycles = 180;

	const float dbMax = 35.0f;
	const float dbMin = -55.0f;

	const float W = 1600.0f;
	const float H = 980.0f;
	const float margin = 20.0f;
	const float gap = 16.0f;

	const float panelW = (W - margin * 2.0f - gap) * 0.5f;
	const float panelH = (H - margin * 2.0f - gap) * 0.5f;

	const auto fcList = GenerateLogFC(fcMin, fcMax, densityPerDecade);

	using IIRDesigner = AnalyticGradient::MatchedComplexIIRDesignAnalytic;

	std::vector<IIRDesigner*> designs[4];
	for (int t = 0; t < numTypes; ++t)
	{
		for (float fc : fcList)
		{
			auto* d = new IIRDesigner(t);
			//d->SetWarpThreshold(warpThresholdHz);
			d->SetupAnalogPrototype(protoType, fc, q, gain, stages);
			d->RunOptimizerDirect(adamCycles, lbfgsCycles);
			designs[t].push_back(d);
		}
	}

	auto DrawSweepPanel =
		[&](float x0, float y0, float x1, float y1, const char* title, const std::vector<IIRDesigner*>& ds)
		{
			DrawPanelBox(x0, y0, x1, y1, title);

			const float gx0 = x0 + 50.0f;
			const float gy0 = y0 + 36.0f;
			const float gx1 = x1 - 20.0f;
			const float gy1 = y1 - 24.0f;

			DrawResponseGrid(gx0, gy0, gx1, gy1, dbMin, dbMax);

			const int nCurves = (int)ds.size();
			for (int i = 0; i < nCurves; ++i)
			{
				const float t = nCurves > 1 ? (float)i / (float)(nCurves - 1) : 0.0f;
				Color c = ColorFromHSV(270.0f * (1.0f - t), 1.0f, 1.0f);

				DrawDesignCurve(ds[i], true, true, gx0, gy0, gx1, gy1, dbMin, dbMax, argb(0x30777777), 1.0f, 384);
				DrawDesignCurve(ds[i], true, false, gx0, gy0, gx1, gy1, dbMin, dbMax, c, 1.8f, 384);
			}

			DrawTextFmt(gx0, y0 + 8, 16.0f, argb(0xffb0b0b0), "fc: %.0f -> %.0f  log  %d/decade", fcMin, fcMax, densityPerDecade);
			DrawTextFmt(gx1 - 220, y0 + 8, 16.0f, argb(0xffb0b0b0), "Adam=%d  Lbfgs=%d", adamCycles, lbfgsCycles);
		};

	while (!WindowShouldClose())
	{
		BeginDrawing();
		ClearBackground(argb(0xff000000));

		const float p00x0 = margin;
		const float p00y0 = margin;
		const float p00x1 = p00x0 + panelW;
		const float p00y1 = p00y0 + panelH;

		const float p01x0 = p00x1 + gap;
		const float p01y0 = margin;
		const float p01x1 = p01x0 + panelW;
		const float p01y1 = p01y0 + panelH;

		const float p10x0 = margin;
		const float p10y0 = p00y1 + gap;
		const float p10x1 = p10x0 + panelW;
		const float p10y1 = p10y0 + panelH;

		const float p11x0 = p10x1 + gap;
		const float p11y0 = p01y1 + gap;
		const float p11x1 = p11x0 + panelW;
		const float p11y1 = p11y0 + panelH;

		DrawSweepPanel(p00x0, p00y0, p00x1, p00y1, typeNames[0], designs[0]);
		DrawSweepPanel(p01x0, p01y0, p01x1, p01y1, typeNames[1], designs[1]);
		DrawSweepPanel(p10x0, p10y0, p10x1, p10y1, typeNames[2], designs[2]);
		DrawSweepPanel(p11x0, p11y0, p11x1, p11y1, typeNames[3], designs[3]);

		DrawTextUI("fc sweep benchmark by parameter space", 20, 4, 20.0f, RAYWHITE);
		DrawTextUI("gray = prototype, rainbow = fitted response", 1160, 4, 17.0f, argb(0xffc0c0c0));

		EndDrawing();
	}

	for (int t = 0; t < numTypes; ++t)
		for (auto* d : designs[t])
			delete d;

	UnloadUIFont();
	CloseWindow();
	return 0;
}

int main()
{
	return TestOptimizationHistory();
	//return TestTopologyBenchmark();
	//return TestFcSweepByParameterSpace();
}