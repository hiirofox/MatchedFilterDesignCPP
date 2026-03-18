#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include "raylib/src/raylib.h"

#include "dsp/optimizer.h"
#include "dsp/filter.h"


class MatchedIIRDesign
{
private:
	constexpr static float M_PI = 3.14159265358979323846f;
	constexpr static int numPoints = 500;

	AdamOptimizer optAdam;
	LbfgsOptimizer3 optLbfgs;
	Optimizer optGrad;
	OptimizerBase* optBase = &optAdam;

	FourStageNonlinearWhiteningIIR iir;
	AnalogPrototypeFilter prototype;

	std::vector<float> coeffs;

	float randNormV()
	{
		const float a = (float)rand() / (float)RAND_MAX;
		const float b = (float)rand() / (float)RAND_MAX;
		return a * b * (rand() % 2 ? 1.0f : -1.0f);
	}

	float spaceTransitionV = 0.2;
	float TransitionSpace(float minv, float maxv, float normx) const
	{
		float logspace = std::exp(std::log(minv) + (std::log(maxv) - std::log(minv)) * normx);
		float linspace = minv + (maxv - minv) * normx;
		return linspace * (1.0 - spaceTransitionV) + logspace * spaceTransitionV;
	}

	float freqSpace[numPoints] = { 0 };
	float magdBSpace[numPoints] = { 0 };
	float magLinSpace[numPoints] = { 0 };
	float meanTargetDB = 0.0f;

	float Error(std::vector<float>& coeffs)
	{
		iir.SetCoeffs(coeffs);

		float totalErrDB = 0.0f;
		for (int i = 0; i < numPoints; ++i)
		{
			const float mag = iir.GetMagResp(freqSpace[i]);
			const float magDB = 20.0f * std::log10(std::max(mag, 1.0e-30f));
			const float errDB = (magDB - magdBSpace[i]);
			//if (magDB < -40 && errDB < -40) continue;
			float e2 = errDB * errDB;
			float e1 = fabsf(errDB);
			float ev = e1 * 0.01 + e2 * 0.99;
			float freqk = freqSpace[i] / 48000.0 * 0.5 + 0.5;
			float dbk = 1.0 / (30.1 + std::min(-30.0f, magdBSpace[i]));
			//totalErrDB += ev * freqk * (0.2 + 0.8 * dbk);
			totalErrDB += ev * freqk;
		}
		return totalErrDB;
	}

public:
	MatchedIIRDesign()
	{
		coeffs.resize(9);
		Init();
	}

	void Init()
	{
		srand(31415926);

		if (coeffs.size() < 9)
			coeffs.resize(9);

		coeffs[0] = randNormV();
		coeffs[1] = randNormV();
		coeffs[2] = randNormV();
		coeffs[3] = randNormV();
		coeffs[4] = randNormV();
		coeffs[5] = randNormV();
		coeffs[6] = randNormV();
		coeffs[7] = randNormV();
		coeffs[8] = randNormV();

		optAdam.SetupOptimizer(9, coeffs, 2);
		optLbfgs.SetupOptimizer(9, coeffs, 0.01f);
		optGrad.SetupOptimizer(9, coeffs, 0.00001f);
		optAdam.SetErrorFunc([this](std::vector<float>& coeffs) { return Error(coeffs); });
		optLbfgs.SetErrorFunc([this](std::vector<float>& coeffs) { return Error(coeffs); });
		optGrad.SetErrorFunc([this](std::vector<float>& coeffs) { return Error(coeffs); });

		for (int i = 0; i < numPoints; ++i)
		{
			freqSpace[i] = TransitionSpace(20.0f, 24000.0f, (float)i / (float)(numPoints - 1));
		}
	}

	void SetupAnalogPrototype(AnalogFilterType type, float fc, float q, float gain, float stages)
	{
		Init();

		for (int i = 0; i < numPoints; ++i)
		{
			const float freqhz = freqSpace[i];
			const float mag = prototype.GetMapResp(type, freqhz, fc, q, gain, stages);
			magLinSpace[i] = std::max(mag, 1.0e-30f);
			magdBSpace[i] = 20.0f * std::log10(std::max(mag, 1.0e-30f));
		}
	}

	int totalCycles = 0;
	int isFirstTimeSwitch = 0;
	void RunOptimizer(int numCycles)
	{
		if (totalCycles > 400)return;

		optBase->RunOptimizer(numCycles);

		totalCycles += numCycles;
		if (totalCycles > 380)
		{
			if (!isFirstTimeSwitch)
			{
				isFirstTimeSwitch = 1;
				optBase->GetBestVec(coeffs);
				optBase = &optLbfgs;
				optBase->SetBasin(coeffs);
			}
		}
	}
	void RunOptimizerDirect()
	{
		optAdam.RunOptimizer(500);
		optAdam.GetBestVec(coeffs);
		optLbfgs.SetBasin(coeffs);
		optLbfgs.RunOptimizer(20);
	}

	void GetNowCoeffs(std::vector<float>& coeffs)
	{
		optBase->GetNowVec(coeffs);
	}
	void GetResponseDB(std::vector<float>& outMagDB, float sampleRate = 48000.0f)
	{
		std::vector<float> nowCoeffs;
		optBase->GetNowVec(nowCoeffs);
		iir.SetCoeffs(nowCoeffs);

		outMagDB.resize(numPoints);
		for (int i = 0; i < numPoints; ++i)
		{
			const float mag = iir.GetMagResp(freqSpace[i], sampleRate);
			outMagDB[i] = 20.0f * std::log10(std::max(mag, 1.0e-30f));
		}
	}
	void GetErrorCurveDB(std::vector<float>& outErrDB, int errMode, float sampleRate = 48000.0f)
	{
		std::vector<float> nowCoeffs;
		optBase->GetNowVec(nowCoeffs);
		iir.SetCoeffs(nowCoeffs);

		outErrDB.resize(numPoints);
		for (int i = 0; i < numPoints; ++i)
		{
			const float fitMag = std::max(iir.GetMagResp(freqSpace[i], sampleRate), 1.0e-30f);

			if (errMode == 0)
			{
				const float fitDB = 20.0f * std::log10(fitMag);
				outErrDB[i] = fitDB - magdBSpace[i];
			}
			else
			{
				const float ratio = fitMag / std::max(magLinSpace[i], 1.0e-30f);
				outErrDB[i] = 20.0f * std::log10(std::max(ratio, 1.0e-30f));
			}
		}
	}

	const float* GetMagDBSpace() const { return magdBSpace; }
	int GetNumPoints()const { return numPoints; }
	float GetFreqAt(int i) const { return freqSpace[i]; }
};

//////////////////////////////////////////////////////////////////////////////////////////////////


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
int main()
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