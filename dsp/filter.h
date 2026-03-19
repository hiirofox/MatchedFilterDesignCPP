#pragma once

#include <complex>
#include <vector>
#include "real_iir_whitening.h"
#include "four_stage_nonlinear_whitening.h"

class IIRFilterBase
{
private:
public:
	virtual void SetCoeffs(const std::vector<float>& coeffs) = 0;
	virtual void GetCoeffs(std::vector<float>& coeffs) = 0;
	virtual float GetMagResp(float freqhz, float sampleRate = 48000) = 0;
};

class TwoStageComplexIIR :public IIRFilterBase
{
private:
	float gain = 1;
	std::complex<float> z1, z2, p1, p2;
public:
	void SetCoeffs(const std::vector<float>& coeffs)
	{
		if (coeffs.size() < 9)return;
		gain = coeffs[0];
		z1 = std::polar(coeffs[1], coeffs[2]);
		z2 = std::polar(coeffs[3], coeffs[4]);
		p1 = std::polar(coeffs[5], coeffs[6]);
		p2 = std::polar(coeffs[7], coeffs[8]);
	}
	void GetCoeffs(std::vector<float>& coeffs)
	{
		coeffs.resize(9);
		coeffs[0] = gain;
		coeffs[1] = std::abs(z1);
		coeffs[2] = std::atan2f(z1.imag(), z1.real());
		coeffs[3] = std::abs(z2);
		coeffs[4] = std::atan2f(z2.imag(), z2.real());
		coeffs[5] = std::abs(p1);
		coeffs[6] = std::atan2f(p1.imag(), p1.real());
		coeffs[7] = std::abs(p2);
		coeffs[8] = std::atan2f(p2.imag(), p2.real());
	}

	float GetMagResp(float freqhz, float sampleRate = 48000)
	{
		const float w = 2.0f * 3.14159265358979323846f * freqhz / sampleRate;
		const std::complex<float> z = std::polar(1.0f, w);
		const std::complex<float> num =
			(z - z1) * (z - std::conj(z1)) *
			(z - z2) * (z - std::conj(z2));
		const std::complex<float> den =
			(z - p1) * (z - std::conj(p1)) *
			(z - p2) * (z - std::conj(p2));
		return std::abs(gain * num / den);
	}
};

class FourStageRealIIR :public IIRFilterBase
{
private:
	float b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0;
public:
	void SetCoeffs(const std::vector<float>& coeffs)
	{
		if (coeffs.size() < 9) return;
		b0 = coeffs[0]; b1 = coeffs[1]; b2 = coeffs[2]; b3 = coeffs[3]; b4 = coeffs[4];
		a1 = coeffs[5]; a2 = coeffs[6]; a3 = coeffs[7]; a4 = coeffs[8];
	}

	void GetCoeffs(std::vector<float>& coeffs)
	{
		coeffs = { b0, b1, b2, b3, b4, a1, a2, a3, a4 };
	}

	float GetMagResp(float freqhz, float sampleRate = 48000.0f)
	{
		const float w = 2.0f * 3.14159265358979323846f * freqhz / sampleRate;
		const std::complex<float> z1 = std::polar(1.0f, -w);
		const std::complex<float> z2 = z1 * z1;
		const std::complex<float> z3 = z2 * z1;
		const std::complex<float> z4 = z2 * z2;

		return std::abs(
			(b0 + b1 * z1 + b2 * z2 + b3 * z3 + b4 * z4) /
			(1.0f + a1 * z1 + a2 * z2 + a3 * z3 + a4 * z4)
		);
	}
};

class TwoStageCosIIR :public IIRFilterBase
{
private:
	float gain = 1.0f;

	float rz1 = 0.0f, thz1 = 0.0f;
	float rz2 = 0.0f, thz2 = 0.0f;
	float rp1 = 0.0f, thp1 = 0.0f;
	float rp2 = 0.0f, thp2 = 0.0f;

	static constexpr float kPI = 3.14159265358979323846f;

	static void ExpandSecondOrder(float r, float th, float& c0, float& c1, float& c2)
	{
		const float a = -2.0f * r * std::cos(th);
		const float b = r * r;

		c0 = 1.0f;
		c1 = a;
		c2 = b;
	}

	static void MultiplyPoly2(
		float a0, float a1, float a2,
		float b0, float b1, float b2,
		float& o0, float& o1, float& o2, float& o3, float& o4)
	{
		o0 = a0 * b0;
		o1 = a0 * b1 + a1 * b0;
		o2 = a0 * b2 + a1 * b1 + a2 * b0;
		o3 = a1 * b2 + a2 * b1;
		o4 = a2 * b2;
	}

public:
	void SetCoeffs(const std::vector<float>& coeffs)
	{
		if (coeffs.size() < 9) return;
		gain = coeffs[0];
		rz1 = coeffs[1];
		thz1 = coeffs[2];
		rz2 = coeffs[3];
		thz2 = coeffs[4];
		rp1 = coeffs[5];
		thp1 = coeffs[6];
		rp2 = coeffs[7];
		thp2 = coeffs[8];
	}

	void GetCoeffs(std::vector<float>& coeffs)
	{
		coeffs.resize(9);
		coeffs[0] = gain;
		coeffs[1] = rz1;
		coeffs[2] = thz1;
		coeffs[3] = rz2;
		coeffs[4] = thz2;
		coeffs[5] = rp1;
		coeffs[6] = thp1;
		coeffs[7] = rp2;
		coeffs[8] = thp2;
	}

	void GetRealIIRCoeffs(std::vector<float>& coeffs) const
	{
		float nz10, nz11, nz12;
		float nz20, nz21, nz22;
		float dp10, dp11, dp12;
		float dp20, dp21, dp22;
		ExpandSecondOrder(rz1, thz1, nz10, nz11, nz12);
		ExpandSecondOrder(rz2, thz2, nz20, nz21, nz22);
		ExpandSecondOrder(rp1, thp1, dp10, dp11, dp12);
		ExpandSecondOrder(rp2, thp2, dp20, dp21, dp22);
		float b0, b1, b2, b3, b4;
		float d0, d1, d2, d3, d4;
		MultiplyPoly2(nz10, nz11, nz12, nz20, nz21, nz22, b0, b1, b2, b3, b4);
		MultiplyPoly2(dp10, dp11, dp12, dp20, dp21, dp22, d0, d1, d2, d3, d4);
		coeffs.resize(9);
		coeffs[0] = gain * b0;
		coeffs[1] = gain * b1;
		coeffs[2] = gain * b2;
		coeffs[3] = gain * b3;
		coeffs[4] = gain * b4;
		// 分母首项恒为 1，这里 d0 理论上就是 1
		coeffs[5] = d1;
		coeffs[6] = d2;
		coeffs[7] = d3;
		coeffs[8] = d4;
	}

	void ToFourStageRealIIR(FourStageRealIIR& outIIR) const
	{
		std::vector<float> coeffs;
		GetRealIIRCoeffs(coeffs);
		outIIR.SetCoeffs(coeffs);
	}

	float GetMagResp(float freqhz, float sampleRate = 48000.0f)
	{
		const float w = 2.0f * kPI * freqhz / sampleRate;
		const float c1 = std::cos(w);
		const float c2 = std::cos(2.0f * w);
		const float s1 = std::sin(w);
		const float s2 = std::sin(2.0f * w);

		std::vector<float> coeffs;
		GetRealIIRCoeffs(coeffs);

		const float b0 = coeffs[0];
		const float b1 = coeffs[1];
		const float b2 = coeffs[2];
		const float b3 = coeffs[3];
		const float b4 = coeffs[4];
		const float a1 = coeffs[5];
		const float a2 = coeffs[6];
		const float a3 = coeffs[7];
		const float a4 = coeffs[8];

		const float nr = b0 + b1 * c1 + b2 * c2 + b3 * std::cos(3.0f * w) + b4 * std::cos(4.0f * w);
		const float ni = -(b1 * s1 + b2 * s2 + b3 * std::sin(3.0f * w) + b4 * std::sin(4.0f * w));

		const float dr = 1.0f + a1 * c1 + a2 * c2 + a3 * std::cos(3.0f * w) + a4 * std::cos(4.0f * w);
		const float di = -(a1 * s1 + a2 * s2 + a3 * std::sin(3.0f * w) + a4 * std::sin(4.0f * w));

		const float num2 = nr * nr + ni * ni;
		const float den2 = dr * dr + di * di;

		return std::sqrt(std::max(num2 / std::max(den2, 1.0e-30f), 1.0e-30f));
	}
};
class FourStageWhiteningIIR :public IIRFilterBase
{
private:
	FourStageRealIIR realIIR;

	std::vector<float> whitenVec;   // v-space
	std::vector<float> realCoeffs;  // a-space : [b0..b4, a1..a4]

public:
	static constexpr int kDim = 9;
private:
	static void MulMat9x9Vec9(const double M[kDim][kDim], const double* x, double* y)
	{
		for (int i = 0; i < kDim; ++i)
		{
			double sum = 0.0;
			for (int j = 0; j < kDim; ++j)
				sum += M[i][j] * x[j];
			y[i] = sum;
		}
	}

	void UpdateRealFromWhiten()
	{
		double v[kDim];
		double a[kDim];

		for (int i = 0; i < kDim; ++i)
			v[i] = (i < (int)whitenVec.size()) ? (double)whitenVec[i] : 0.0;

		MulMat9x9Vec9(kWhitenRInv, v, a);

		realCoeffs.resize(kDim);
		for (int i = 0; i < kDim; ++i)
			realCoeffs[i] = (float)(a[i] + kWhitenMu[i]);

		realIIR.SetCoeffs(realCoeffs);
	}

	void UpdateWhitenFromReal()
	{
		double x[kDim];
		double v[kDim];

		for (int i = 0; i < kDim; ++i)
			x[i] = ((i < (int)realCoeffs.size()) ? (double)realCoeffs[i] : 0.0) - kWhitenMu[i];

		MulMat9x9Vec9(kWhitenR, x, v);

		whitenVec.resize(kDim);
		for (int i = 0; i < kDim; ++i)
			whitenVec[i] = (float)v[i];
	}

public:
	FourStageWhiteningIIR()
	{
		whitenVec.assign(kDim, 0.0f);
		realCoeffs.assign(kDim, 0.0f);
		UpdateRealFromWhiten();
	}

	// ============================================================
	// 对外默认使用 whiten space
	// ============================================================

	void SetCoeffs(const std::vector<float>& coeffs)
	{
		whitenVec = coeffs;
		whitenVec.resize(kDim, 0.0f);
		UpdateRealFromWhiten();
	}

	void GetCoeffs(std::vector<float>& coeffs)
	{
		coeffs = whitenVec;
	}

	// ============================================================
	// 显式访问 real iir coeffs
	// ============================================================

	void SetRealCoeffs(const std::vector<float>& coeffs)
	{
		realCoeffs = coeffs;
		realCoeffs.resize(kDim, 0.0f);
		realIIR.SetCoeffs(realCoeffs);
		UpdateWhitenFromReal();
	}

	void GetRealCoeffs(std::vector<float>& coeffs) const
	{
		coeffs = realCoeffs;
	}

	// ============================================================
	// 频响
	// ============================================================

	float GetMagResp(float freqhz, float sampleRate = 48000.0f)
	{
		return realIIR.GetMagResp(freqhz, sampleRate);
	}

	// ============================================================
	// 静态变换接口，方便外部直接用
	// ============================================================

	static void RealToWhiten(const std::vector<float>& realIn, std::vector<float>& whitenOut)
	{
		double x[kDim];
		double v[kDim];

		for (int i = 0; i < kDim; ++i)
			x[i] = ((i < (int)realIn.size()) ? (double)realIn[i] : 0.0) - kWhitenMu[i];

		MulMat9x9Vec9(kWhitenR, x, v);

		whitenOut.resize(kDim);
		for (int i = 0; i < kDim; ++i)
			whitenOut[i] = (float)v[i];
	}

	static void WhitenToReal(const std::vector<float>& whitenIn, std::vector<float>& realOut)
	{
		double v[kDim];
		double a[kDim];

		for (int i = 0; i < kDim; ++i)
			v[i] = (i < (int)whitenIn.size()) ? (double)whitenIn[i] : 0.0;

		MulMat9x9Vec9(kWhitenRInv, v, a);

		realOut.resize(kDim);
		for (int i = 0; i < kDim; ++i)
			realOut[i] = (float)(a[i] + kWhitenMu[i]);
	}
};
class FourStageNonlinearWhiteningIIR :public IIRFilterBase
{
private:
	static constexpr int kDim = 9;

	FourStageRealIIR realIIR;

	std::vector<float> vCoeffs;   // 非线性 whitening 空间
	std::vector<float> aCoeffs;   // 真实4阶实系数空间 [b0..b4, a1..a4]

	void UpdateRealFromV()
	{
		vCoeffs.resize(kDim, 0.0f);
		FourStageNonlinearWhitening::ForwardVToA(vCoeffs, aCoeffs);
		aCoeffs.resize(kDim, 0.0f);
		realIIR.SetCoeffs(aCoeffs);
	}

	void UpdateVFromReal()
	{
		aCoeffs.resize(kDim, 0.0f);
		realIIR.SetCoeffs(aCoeffs);
		FourStageNonlinearWhitening::InverseAToV(aCoeffs, vCoeffs);
		vCoeffs.resize(kDim, 0.0f);
	}

public:
	FourStageNonlinearWhiteningIIR()
	{
		vCoeffs.assign(kDim, 0.0f);
		aCoeffs.assign(kDim, 0.0f);
		UpdateRealFromV();
	}

	// ============================================================
	// 与你原来的接口风格保持一致：
	// SetCoeffs / GetCoeffs 默认操作的是 whitening 空间
	// ============================================================

	void SetCoeffs(const std::vector<float>& coeffs)
	{
		vCoeffs = coeffs;
		vCoeffs.resize(kDim, 0.0f);
		UpdateRealFromV();
	}

	void GetCoeffs(std::vector<float>& coeffs)
	{
		coeffs = vCoeffs;
	}

	// ============================================================
	// 调试/导出用：直接访问 real IIR coeffs
	// ============================================================

	void SetRealCoeffs(const std::vector<float>& coeffs)
	{
		aCoeffs = coeffs;
		aCoeffs.resize(kDim, 0.0f);
		realIIR.SetCoeffs(aCoeffs);
		UpdateVFromReal();
	}

	void GetRealCoeffs(std::vector<float>& coeffs) const
	{
		coeffs = aCoeffs;
	}

	// ============================================================
	// 频响
	// ============================================================

	float GetMagResp(float freqhz, float sampleRate = 48000.0f)
	{
		return realIIR.GetMagResp(freqhz, sampleRate);
	}

	// ============================================================
	// 直接暴露静态变换，方便外部单独测试
	// ============================================================

	static void VToReal(const std::vector<float>& vin, std::vector<float>& aout)
	{
		FourStageNonlinearWhitening::ForwardVToA(vin, aout);
	}

	static void RealToV(const std::vector<float>& ain, std::vector<float>& vout)
	{
		FourStageNonlinearWhitening::InverseAToV(ain, vout);
	}
};

///////////////////////////////////////////////////////////////////////

enum class AnalogFilterType
{
	LP, BP, HP, Peaking, LS, HS
};

class AnalogPrototypeFilter
{
private:
	constexpr static float M_PI = 3.14159265358979323846f;
public:
	float GetMapResp(AnalogFilterType type, float freqhz, float fc, float q, float gain, float stages)
	{
		const float wc = 2.0f * M_PI * fc;
		const std::complex<float> s(0.0f, 2.0f * M_PI * freqhz);
		const float w = std::abs(s.imag());

		constexpr float eps30 = 1.0e-30f;
		constexpr float eps200 = 1.0e-30f;

		switch (type)
		{
		case AnalogFilterType::Peaking:
		{
			const float A_target = std::pow(10.0f, gain / 20.0f);
			const float B = wc / q;
			const float x = std::abs((w * w - wc * wc) / (B * w + eps200));
			const float W = 1.0f / (1.0f + std::pow(x, 2.0f * stages));
			const float mag = 1.0f + (A_target - 1.0f) * W;
			return mag;
		}

		case AnalogFilterType::LP:
		{
			float peak_factor;
			if (q > 1.0f / std::sqrt(2.0f))
				peak_factor = 1.0f - 1.0f / (2.0f * q * q);
			else
				peak_factor = 1.0f;

			const float wc_comp = wc * std::pow(peak_factor, 0.5f - 1.0f / (2.0f * stages));
			const float y = w / (wc_comp + eps30);

			const std::complex<float> denominator(
				1.0f - std::pow(y, 2.0f * stages),
				std::pow(y, stages) / q
			);

			const std::complex<float> h = 1.0f / denominator;
			return std::abs(h);
		}

		case AnalogFilterType::HP:
		{
			float peak_factor;
			if (q > 1.0f / std::sqrt(2.0f))
				peak_factor = 1.0f - 1.0f / (2.0f * q * q);
			else
				peak_factor = 1.0f;

			const float wc_comp = wc * std::pow(peak_factor, 0.5f - 1.0f / (2.0f * stages));
			const float y = w / (wc_comp + eps30);

			const float numerator = std::pow(y, 2.0f * stages);
			const std::complex<float> denominator(
				1.0f - std::pow(y, 2.0f * stages),
				std::pow(y, stages) / q
			);

			const std::complex<float> h = numerator / (denominator + std::complex<float>(eps30, 0.0f));
			return std::abs(h);
		}

		case AnalogFilterType::BP:
		{
			const float y = w / (wc + eps30);
			const float x = std::abs((y * y - 1.0f) / ((y / q) + eps200));
			const float W = 1.0f / (1.0f + std::pow(x, 2.0f * stages));
			return W;
		}

		case AnalogFilterType::LS:
		{
			const float V = std::pow(10.0f, gain / 20.0f);
			const float x = w / (wc + eps30);
			const float beta = (1.0f / (q * q)) - 2.0f;

			const float x_N = std::pow(x, stages);
			const float x_2N = std::pow(x, 2.0f * stages);
			const float mid_term = std::sqrt(V) * beta * x_N;

			const float num = V + mid_term + x_2N;
			const float den = 1.0f + mid_term + V * x_2N;
			const float mag_sq = V * (num / den);
			const float mag = std::sqrt(std::fmax(mag_sq, 0.0f));
			return mag;
		}

		case AnalogFilterType::HS:
		{
			const float V = std::pow(10.0f, gain / 20.0f);
			const float x = w / (wc + eps30);
			const float beta = (1.0f / (q * q)) - 2.0f;

			const float x_N = std::pow(x, stages);
			const float x_2N = std::pow(x, 2.0f * stages);
			const float mid_term = std::sqrt(V) * beta * x_N;

			const float num = 1.0f + mid_term + V * x_2N;
			const float den = V + mid_term + x_2N;
			const float mag_sq = V * (num / den);
			const float mag = std::sqrt(std::fmax(mag_sq, 0.0f));
			return mag;
		}

		default:
			return 1.0f;
		}
	}
};

///////////////////////////////////////////////////////////////////////////////
class IIRDesignBase
{
private:
public:
	//计算用
	virtual void Init() = 0;
	virtual void SetupAnalogPrototype(AnalogFilterType type, float fc, float q, float gain, float stages) = 0;
	virtual void RunOptimizer(int numCycles, int maxCycles) = 0;
	virtual void RunOptimizerDirect(int adamCycles = 40, int lbfgsCycles = 160) = 0;
	virtual void GetNowCoeffs(std::vector<float>& coeffs) = 0;
	virtual void GetBestCoeffs(std::vector<float>& coeffs) = 0;

	//绘制用
	virtual float GetPrototypeResp(float freqhz) = 0;
	virtual float GetNowIIRResp(float freqhz) = 0;
	virtual float GetBestIIRResp(float freqhz) = 0;
};

class MatchedIIRDesign :public IIRDesignBase
{
private:
	constexpr static float M_PI = 3.14159265358979323846f;
	constexpr static int numPoints = 100;

	AdamOptimizer optAdam;
	LbfgsOptimizerLightweight optLbfgs;
	//Optimizer optGrad;
	OptimizerBase* optBase = &optAdam;

	//best coeffs:
	//FourStageNonlinearWhiteningIIR	lr=2.0
	//FourStageRealIIR					lr=0.01
	//TwoStageComplexIIR				lr=0.25
	//TwoStageCosIIR					lr=0.35
	int selectIIRType = 0;
	IIRFilterBase* iirs[4] = { new FourStageNonlinearWhiteningIIR,new FourStageRealIIR,new TwoStageComplexIIR,new TwoStageCosIIR };
	float adamLearningRate[4] = { 2.0,0.01,0.25,0.350 };

	IIRFilterBase* iir = iirs[selectIIRType];
	AnalogPrototypeFilter prototype;

	std::vector<float> coeffs;

	float randNormV()
	{
		const float a = (float)rand() / (float)RAND_MAX;
		const float b = (float)rand() / (float)RAND_MAX;
		return a * b * (rand() % 2 ? 1.0f : -1.0f);
	}

	float spaceTransitionV = 0.5;
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

	float Error(const std::vector<float>& coeffs) const
	{
		iir->SetCoeffs(coeffs);

		float totalErrDB = 0.0f;
		for (int i = 0; i < numPoints; ++i)
		{
			const float mag = iir->GetMagResp(freqSpace[i]);
			const float magDB = 20.0f * std::log10(std::max(mag, 1.0e-30f));
			const float errDB = (magDB - magdBSpace[i]);
			//if (magDB < -40 && errDB < -40) continue;
			float e2 = errDB * errDB;
			float e1 = fabsf(errDB);
			float ev = e1 * 0.001 + e2 * 0.999;
			float freqk = freqSpace[i] / 48000.0 * 0.5 + 0.5;
			float dbk = 1.0 / (30.1 + std::min(-30.0f, magdBSpace[i]));
			//totalErrDB += ev * freqk * (0.2 + 0.8 * dbk);
			totalErrDB += ev * freqk;
		}
		return totalErrDB;
	}

public:
	MatchedIIRDesign(int iirtype = 2)
	{
		if (iirtype > 3)iirtype = 3;
		if (iirtype < 0)iirtype = 0;

		selectIIRType = iirtype;
		iir = iirs[selectIIRType];
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

		optAdam.SetupOptimizer(9, coeffs, adamLearningRate[selectIIRType]);
		optLbfgs.SetupOptimizer(9, coeffs, 0.01f);
		//optGrad.SetupOptimizer(9, coeffs, 0.00001f);
		optAdam.SetErrorFunc([this](std::vector<float>& coeffs) { return Error(coeffs); });
		optLbfgs.SetErrorFunc([this](std::vector<float>& coeffs) { return Error(coeffs); });
		//optGrad.SetErrorFunc([this](std::vector<float>& coeffs) { return Error(coeffs); });

		for (int i = 0; i < numPoints; ++i)
		{
			freqSpace[i] = TransitionSpace(20.0f, 24000.0f, (float)i / (float)(numPoints - 1));
		}
	}

	AnalogFilterType now_type = AnalogFilterType::LP;
	float now_fc = 5000;
	float now_q = 0.707;
	float now_gain = 10;
	float now_stages = 2;
	void SetupAnalogPrototype(AnalogFilterType type, float fc, float q, float gain, float stages)override
	{
		Init();

		now_type = type;
		now_fc = fc;
		now_q = q;
		now_gain = gain;
		now_stages = stages;

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
	void RunOptimizer(int numCycles, int maxCycles)override
	{
		if (totalCycles > maxCycles)return;

		optBase->RunOptimizer(numCycles);

		totalCycles += numCycles;
		if (totalCycles > maxCycles * 4.0 / 10.0)
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
	void RunOptimizerDirect(int adamCycles = 40, int lbfgsCycles = 160)override//建议用这个
	{
		optAdam.RunOptimizer(adamCycles);
		optAdam.GetBestVec(coeffs);
		optLbfgs.SetBasin(coeffs);
		optLbfgs.RunOptimizer(lbfgsCycles);
		optBase = &optLbfgs;
	}
	void GetNowCoeffs(std::vector<float>& coeffs)override
	{
		optBase->GetNowVec(coeffs);
	}
	void GetBestCoeffs(std::vector<float>& coeffs)override
	{
		optBase->GetBestVec(coeffs);
	}

	//绘制频响用：
	float GetPrototypeResp(float freqhz)override
	{
		return prototype.GetMapResp(now_type, freqhz, now_fc, now_q, now_gain, now_stages);
	}
	float GetNowIIRResp(float freqhz)override
	{
		optBase->GetNowVec(coeffs);
		iir->SetCoeffs(coeffs);
		return iir->GetMagResp(freqhz);
	}
	float GetBestIIRResp(float freqhz)override//一般来说用这个
	{
		optBase->GetBestVec(coeffs);
		iir->SetCoeffs(coeffs);
		return iir->GetMagResp(freqhz);
	}
};

class WarpedMatchedIIRDesign :public IIRDesignBase
{
private:
	constexpr static float M_PI = 3.14159265358979323846f;
	constexpr static int numPoints = 50;
	constexpr static float kSampleRate = 48000.0f;

	AdamOptimizer optAdam;
	LbfgsOptimizerLightweight optLbfgs;
	OptimizerBase* optBase = &optAdam;

	int selectIIRType = 0;
	IIRFilterBase* iirs[4] =
	{
		new FourStageNonlinearWhiteningIIR,
		new FourStageRealIIR,
		new TwoStageComplexIIR,
		new TwoStageCosIIR
	};

	float adamLearningRate[4] = { 2.0f,0.01f,0.25f,0.35f };

	IIRFilterBase* iir = iirs[0];
	AnalogPrototypeFilter prototype;

	std::vector<float> coeffs;

	// -------------------------
	// warp state
	// -------------------------
	float warpThreshold = 15000.0f;
	bool warpEnabled = false;
	float warpA = 0.0f;

	AnalogFilterType lastType = AnalogFilterType::LP;
	float lastFc = 1000.0f;
	float lastQ = 0.707f;
	float lastGain = 0.0f;
	float lastStages = 1.0f;

	// display domain: 用户看到的原始频率轴
	float displayFreqSpace[numPoints] = { 0 };
	float displayMagDBSpace[numPoints] = { 0 };
	float displayMagLinSpace[numPoints] = { 0 };

	// fit domain: 真正用于优化的目标轴
	float fitFreqSpace[numPoints] = { 0 };
	float fitMagDBSpace[numPoints] = { 0 };
	float fitMagLinSpace[numPoints] = { 0 };

private:
	float randNormV()
	{
		const float a = (float)rand() / (float)RAND_MAX;
		const float b = (float)rand() / (float)RAND_MAX;
		return a * b * (rand() % 2 ? 1.0f : -1.0f);
	}

	float spaceTransitionV = 0.3f;
	float TransitionSpace(float minv, float maxv, float normx) const
	{
		float logspace = std::exp(std::log(minv) + (std::log(maxv) - std::log(minv)) * normx);
		float linspace = minv + (maxv - minv) * normx;
		return linspace * (1.0f - spaceTransitionV) + logspace * spaceTransitionV;
	}

	static float Clamp(float x, float lo, float hi)
	{
		return (x < lo) ? lo : ((x > hi) ? hi : x);
	}

	static float WarpFreqRad(float w, float a)
	{
		const float num = (1.0f - a * a) * std::sin(w);
		const float den = (1.0f + a * a) * std::cos(w) - 2.0f * a;
		float wp = std::atan2(num, den);
		if (wp < 0.0f) wp += (float)M_PI;
		return Clamp(wp, 0.0f, (float)M_PI);
	}

	static float WarpFreqHz(float fHz, float a, float sampleRate = kSampleRate)
	{
		const float w = 2.0f * (float)M_PI * fHz / sampleRate;
		const float wp = WarpFreqRad(Clamp(w, 1.0e-9f, (float)M_PI - 1.0e-9f), a);
		return wp * sampleRate / (2.0f * (float)M_PI);
	}

	static float SolveWarpA(float fc, float threshold, float sampleRate = kSampleRate)
	{
		if (fc >= threshold)
			return 0.0f;

		const float wc = Clamp(2.0f * (float)M_PI * fc / sampleRate, 1.0e-9f, (float)M_PI - 1.0e-9f);
		const float wt = Clamp(2.0f * (float)M_PI * threshold / sampleRate, 1.0e-9f, (float)M_PI - 1.0e-9f);

		auto func = [&](float a)
			{
				return WarpFreqRad(wc, a) - wt;
			};

		float lo = 0.0f;
		float hi = 0.999999f;
		float flo = func(lo);
		float fhi = func(hi);

		if (flo == 0.0f) return lo;
		if (fhi == 0.0f) return hi;
		if (flo * fhi > 0.0f)
			return 0.0f;

		for (int iter = 0; iter < 80; ++iter)
		{
			const float mid = 0.5f * (lo + hi);
			const float fmid = func(mid);

			if (flo * fmid <= 0.0f)
			{
				hi = mid;
				fhi = fmid;
			}
			else
			{
				lo = mid;
				flo = fmid;
			}
		}

		return 0.5f * (lo + hi);
	}



	float Error(const std::vector<float>& coeffs) const
	{
		iir->SetCoeffs(coeffs);

		float totalErrDB = 0.0f;
		for (int i = 0; i < numPoints; ++i)
		{
			const float mag = iir->GetMagResp(fitFreqSpace[i]);
			const float magDB = 20.0f * std::log10(std::max(mag, 1.0e-30f));
			const float errDB = (magDB - fitMagDBSpace[i]);
			//if (magDB < -40 && errDB < -40) continue;
			float e2 = errDB * errDB;
			float e1 = fabsf(errDB);
			float ev = e1 * 0.01 + e2 * 0.99;
			float freqk = fitFreqSpace[i] / 48000.0 * 0.5 + 0.5;
			float dbk = 1.0 / (30.1 + std::min(-30.0f, fitMagDBSpace[i]));
			//totalErrDB += ev * freqk * (0.2 + 0.8 * dbk);
			totalErrDB += ev * freqk;
		}
		return totalErrDB / numPoints * 100.0;
	}

	void RebuildTargetSpaces()
	{
		// display 轴始终是原始观察轴
		for (int i = 0; i < numPoints; ++i)
		{
			displayFreqSpace[i] = TransitionSpace(20.0f, 24000.0f, (float)i / (float)(numPoints - 1));
		}

		warpEnabled = (lastFc < warpThreshold);
		warpA = warpEnabled ? SolveWarpA(lastFc, warpThreshold, kSampleRate) : 0.0f;

		for (int i = 0; i < numPoints; ++i)
		{
			// 设计频率轴本身仍然用标准网格
			fitFreqSpace[i] = displayFreqSpace[i];

			// display domain 的目标：原始原型
			{
				const float f = displayFreqSpace[i];
				const float mag = prototype.GetMapResp(lastType, f, lastFc, lastQ, lastGain, lastStages);
				displayMagLinSpace[i] = std::max(mag, 1.0e-30f);
				displayMagDBSpace[i] = 20.0f * std::log10(displayMagLinSpace[i]);
			}

			// fit domain 的目标：
			// 若启用 warp，则在设计频率 f' 上，目标取自原始轴 inverse-warp 后的位置
			{
				float srcFreq = fitFreqSpace[i];
				if (warpEnabled)
					srcFreq = WarpFreqHz(fitFreqSpace[i], -warpA, kSampleRate);

				const float mag = prototype.GetMapResp(lastType, srcFreq, lastFc, lastQ, lastGain, lastStages);
				fitMagLinSpace[i] = std::max(mag, 1.0e-30f);
				fitMagDBSpace[i] = 20.0f * std::log10(fitMagLinSpace[i]);
			}
		}
	}

public:
	WarpedMatchedIIRDesign(int iirtype = 2)
	{
		if (iirtype > 3) iirtype = 3;
		if (iirtype < 0) iirtype = 0;

		selectIIRType = iirtype;
		iir = iirs[selectIIRType];
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

		optAdam.SetupOptimizer(9, coeffs, adamLearningRate[selectIIRType] * 1.5);
		optLbfgs.SetupOptimizer(9, coeffs, 0.5f);

		optAdam.SetErrorFunc([this](std::vector<float>& v) { return Error(v); });
		optLbfgs.SetErrorFunc([this](std::vector<float>& v) { return Error(v); });

		for (int i = 0; i < numPoints; ++i)
		{
			displayFreqSpace[i] = TransitionSpace(20.0f, 24000.0f, (float)i / (float)(numPoints - 1));
			fitFreqSpace[i] = displayFreqSpace[i];
			displayMagDBSpace[i] = 0.0f;
			displayMagLinSpace[i] = 1.0f;
			fitMagDBSpace[i] = 0.0f;
			fitMagLinSpace[i] = 1.0f;
		}
	}

	void SetWarpThreshold(float hz)
	{
		warpThreshold = hz;
	}

	float GetWarpThreshold() const
	{
		return warpThreshold;
	}

	bool IsWarpEnabled() const
	{
		return warpEnabled;
	}

	float GetWarpA() const
	{
		return warpA;
	}

	void SetupAnalogPrototype(AnalogFilterType type, float fc, float q, float gain, float stages)override
	{
		Init();

		lastType = type;
		lastFc = fc;
		lastQ = q;
		lastGain = gain;
		lastStages = stages;

		RebuildTargetSpaces();
	}

	int totalCycles = 0;
	int isFirstTimeSwitch = 0;

	void RunOptimizer(int numCycles, int maxCycles)override
	{
		if (totalCycles > maxCycles) return;

		optBase->RunOptimizer(numCycles);

		totalCycles += numCycles;
		if (totalCycles > maxCycles * 0.5f / 10.0f)
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

	void RunOptimizerDirect(int adamCycles = 40, int lbfgsCycles = 160)override
	{
		optAdam.RunOptimizer(adamCycles);
		optAdam.GetBestVec(coeffs);
		optLbfgs.SetBasin(coeffs);
		optLbfgs.RunOptimizer(lbfgsCycles);
		optBase = &optLbfgs;
	}

	void GetNowCoeffs(std::vector<float>& outCoeffs)override
	{
		optBase->GetNowVec(outCoeffs);
	}
	void GetBestCoeffs(std::vector<float>& outCoeffs) override
	{
		optBase->GetBestVec(outCoeffs);
	}

	//绘制频响用：
	float GetPrototypeResp(float freqhz) override
	{
		return prototype.GetMapResp(lastType, freqhz, lastFc, lastQ, lastGain, lastStages);
	}

	float GetNowIIRResp(float freqhz) override
	{
		optBase->GetNowVec(coeffs);
		iir->SetCoeffs(coeffs);
		float evalFreq = freqhz;
		if (warpEnabled)
			evalFreq = WarpFreqHz(freqhz, warpA, kSampleRate);
		return iir->GetMagResp(evalFreq, kSampleRate);
	}

	float GetBestIIRResp(float freqhz) override
	{
		optBase->GetBestVec(coeffs);
		iir->SetCoeffs(coeffs);
		float evalFreq = freqhz;
		if (warpEnabled)
			evalFreq = WarpFreqHz(freqhz, warpA, kSampleRate);
		return iir->GetMagResp(evalFreq, kSampleRate);
	}
};