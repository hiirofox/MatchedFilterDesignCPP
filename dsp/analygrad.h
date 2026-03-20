#pragma once

#include <vector>
#include <functional>
#include <cmath>

#include "filter.h"

namespace AnalyticGradient
{

	struct MagErrorBase
	{
		std::vector<double> globalDB;
		std::vector<double> freqSpace;

		void SetupMagLinearGlobal(std::vector<double>& maglinear, std::vector<double>& freqSpaceHz, int numPoints)
		{
			globalDB.resize(maglinear.size());
			for (int i = 0; i < numPoints; ++i)
			{
				globalDB[i] = 20.0 * log10f(maglinear[i]);
			}
			freqSpace = freqSpaceHz;
		}

		const std::vector<double>& GetFreqSpace() const { return freqSpace; }
		const std::vector<double>& GetGlobalDB() const { return globalDB; }

		virtual double Error(int i, double freqhz, double iirmagdb)
		{
			const double e = iirmagdb - globalDB[i];
			return e * e + 1e-4;
		}

		// 对 Error(i, freqhz, iirmagdb) 关于 iirmagdb 的数值微分
		virtual double dError_dMagDb(int i, double freqhz, double iirmagdb)
		{
			// 中心差分步长，单位 dB
			const double h = 1.0e-4;
			const double ep = Error(i, freqhz, iirmagdb + h);
			const double em = Error(i, freqhz, iirmagdb - h);
			return (ep - em) / (2.0 * h);
		}

		virtual ~MagErrorBase() = default;
	};

	template<int numOrders = 2> // numOrders个零点，numOrders个极点
	class ComplexIIRGradient
	{
	private:
		static constexpr double kPi = 3.1415926535897932384626433832795;
		static constexpr double kDbScale = 20.0 / 2.3025850929940456840179914546844; // 20 / ln(10)
		static constexpr double kTiny = 1.0e-18;

	public:
		// 参数向量安排：gaindB, z0re, z0im, ..., p0re, p0im, ...
		void CalcGradient(std::vector<double>& paramsIn,
			std::vector<double>& paramsGradOut,
			MagErrorBase& magErrMethod,
			double sampleRate = 48000.0)
		{
			const int expectedNumParams = 1 + 4 * numOrders;
			////assert((int)paramsIn.size() == expectedNumParams);

			const std::vector<double>& freqHzs = magErrMethod.GetFreqSpace();
			const int numPoints = (int)freqHzs.size();

			paramsGradOut.assign(expectedNumParams, 0.0);

			const double gainDb = paramsIn[0];

			for (int k = 0; k < numPoints; ++k)
			{
				const double freqHz = freqHzs[k];
				const double w = 2.0 * kPi * freqHz / sampleRate;
				const double cw = cos(w);
				const double sw = sin(w);

				// 先算当前频点的 IIR log-magnitude in dB
				double magDbA = gainDb;

				// zero contributions
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * i;
					const double zr = paramsIn[base + 0];
					const double zi = paramsIn[base + 1];

					const double dr = cw - zr;
					const double di = sw - zi;
					const double d2 = dr * dr + di * di + kTiny;

					magDbA += 10.0 * log10f(d2);
				}

				// pole contributions
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * numOrders + 2 * i;
					const double pr = paramsIn[base + 0];
					const double pi = paramsIn[base + 1];

					const double dr = cw - pr;
					const double di = sw - pi;
					const double d2 = dr * dr + di * di + kTiny;

					magDbA -= 10.0 * log10f(d2);
				}

				// 单频点损失对 magDbA 的导数
				const double psi = magErrMethod.dError_dMagDb(k, freqHz, magDbA);

				// 对 gaindB 的导数恒为 1
				paramsGradOut[0] += psi;

				// 对零点参数的导数
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * i;
					const double zr = paramsIn[base + 0];
					const double zi = paramsIn[base + 1];

					const double dx = zr - cw;
					const double dy = zi - sw;
					const double d2 = dx * dx + dy * dy + kTiny;

					double dMagDb_dZr = kDbScale * dx / d2;
					double dMagDb_dZi = kDbScale * dy / d2;

					// 可选：单位圆附近梯度限制/钳制
					// {
					//     const double mag = std::sqrt(zr * zr + zi * zi);
					//     const double magMax = 0.999;
					//     if (mag > magMax && mag < 1.0 / magMax)
					//     {
					//         const double gradClamp = 1.0e3;
					//         dMagDb_dZr = std::clamp(dMagDb_dZr, -gradClamp, gradClamp);
					//         dMagDb_dZi = std::clamp(dMagDb_dZi, -gradClamp, gradClamp);
					//     }
					// }

					paramsGradOut[base + 0] += psi * dMagDb_dZr;
					paramsGradOut[base + 1] += psi * dMagDb_dZi;
				}

				// 对极点参数的导数
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * numOrders + 2 * i;
					const double pr = paramsIn[base + 0];
					const double pi = paramsIn[base + 1];

					const double dx = pr - cw;
					const double dy = pi - sw;
					const double d2 = dx * dx + dy * dy + kTiny;

					double dMagDb_dPr = -kDbScale * dx / d2;
					double dMagDb_dPi = -kDbScale * dy / d2;

					// 可选：单位圆附近梯度限制/钳制
					// {
					//     const double mag = std::sqrt(pr * pr + pi * pi);
					//     const double magMax = 0.999;
					//     if (mag > magMax && mag < 1.0 / magMax)
					//     {
					//         const double gradClamp = 1.0e3;
					//         dMagDb_dPr = std::clamp(dMagDb_dPr, -gradClamp, gradClamp);
					//         dMagDb_dPi = std::clamp(dMagDb_dPi, -gradClamp, gradClamp);
					//     }
					// }

					paramsGradOut[base + 0] += psi * dMagDb_dPr;
					paramsGradOut[base + 1] += psi * dMagDb_dPi;
				}
			}
		}
		double CalcTotalLoss(const std::vector<double>& paramsIn,
			MagErrorBase& magErrMethod,
			double sampleRate = 48000.0)
		{
			const int expectedNumParams = 1 + 4 * numOrders;
			////assert((int)paramsIn.size() == expectedNumParams);

			const std::vector<double>& freqHzs = magErrMethod.GetFreqSpace();
			const std::vector<double>& targetDB = magErrMethod.GetGlobalDB();

			const int numPoints = (int)freqHzs.size();
			//////assert((int)targetDB.size() == numPoints);

			const double gainDb = paramsIn[0];
			double totalLoss = 0.0;

			for (int k = 0; k < numPoints; ++k)
			{
				const double freqHz = freqHzs[k];
				const double w = 2.0 * kPi * freqHz / sampleRate;
				const double cw = cos(w);
				const double sw = sin(w);
				double magDbA = gainDb;
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * i;
					const double zr = paramsIn[base + 0];
					const double zi = paramsIn[base + 1];

					const double dr = cw - zr;
					const double di = sw - zi;
					const double d2 = dr * dr + di * di + kTiny;

					magDbA += 10.0 * log10(d2);
				}
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * numOrders + 2 * i;
					const double pr = paramsIn[base + 0];
					const double pi = paramsIn[base + 1];

					const double dr = cw - pr;
					const double di = sw - pi;
					const double d2 = dr * dr + di * di + kTiny;

					magDbA -= 10.0 * log10(d2);
				}
				totalLoss += magErrMethod.Error(k, freqHz, magDbA);
			}
			return totalLoss;
		}

		double GetMagResp(const std::vector<double>& params, double freqHz, double sampleRate = 48000.0)
		{
			const int expectedNumParams = 1 + 4 * numOrders;
			//////assert((int)params.size() == expectedNumParams);

			const double gainDb = params[0];

			const double w = 2.0 * kPi * freqHz / sampleRate;
			const double cw = std::cos(w);
			const double sw = std::sin(w);

			double logMagDb = gainDb;

			// zeros
			for (int i = 0; i < numOrders; ++i)
			{
				const int base = 1 + 2 * i;
				const double zr = params[base + 0];
				const double zi = params[base + 1];

				const double dr = cw - zr;
				const double di = sw - zi;
				const double d2 = dr * dr + di * di + kTiny;

				logMagDb += 10.0 * std::log10(d2);
			}

			// poles
			for (int i = 0; i < numOrders; ++i)
			{
				const int base = 1 + 2 * numOrders + 2 * i;
				const double pr = params[base + 0];
				const double pi = params[base + 1];

				const double dr = cw - pr;
				const double di = sw - pi;
				const double d2 = dr * dr + di * di + kTiny;

				logMagDb -= 10.0 * std::log10(d2);
			}

			// dB -> linear
			const double magLinear = std::pow(10.0, logMagDb / 20.0);
			return magLinear;
		}
	};
	template<int numOrders = 2> // numOrders个零点，numOrders个极点
	class ComplexIIRGradientMagTheta
	{
	private:
		static constexpr double kPi = 3.1415926535897932384626433832795;
		static constexpr double kDbScale = 20.0 / 2.3025850929940456840179914546844; // 20 / ln(10)
		static constexpr double kTiny = 1.0e-18;

	public:
		// 参数向量安排：gaindB, z0mag, z0theta, ..., p0mag, p0theta, ...
		void CalcGradient(std::vector<double>& paramsIn,
			std::vector<double>& paramsGradOut,
			MagErrorBase& magErrMethod,
			double sampleRate = 48000.0)
		{
			const int expectedNumParams = 1 + 4 * numOrders;
			//////assert((int)paramsIn.size() == expectedNumParams);

			const std::vector<double>& freqHzs = magErrMethod.GetFreqSpace();
			const int numPoints = (int)freqHzs.size();

			paramsGradOut.assign(expectedNumParams, 0.0);

			const double gainDb = paramsIn[0];

			for (int k = 0; k < numPoints; ++k)
			{
				const double freqHz = freqHzs[k];
				const double w = 2.0 * kPi * freqHz / sampleRate;

				double magDbA = gainDb;

				// zero contributions
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * i;
					const double zm = paramsIn[base + 0];
					const double zt = paramsIn[base + 1];

					const double delta = w - zt;
					const double cosDelta = std::cos(delta);
					const double d2 = 1.0 + zm * zm - 2.0 * zm * cosDelta + kTiny;

					magDbA += 10.0 * std::log10(d2);
				}

				// pole contributions
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * numOrders + 2 * i;
					const double pm = paramsIn[base + 0];
					const double pt = paramsIn[base + 1];

					const double delta = w - pt;
					const double cosDelta = std::cos(delta);
					const double d2 = 1.0 + pm * pm - 2.0 * pm * cosDelta + kTiny;

					magDbA -= 10.0 * std::log10(d2);
				}

				// 单频点损失对 magDbA 的导数
				const double psi = magErrMethod.dError_dMagDb(k, freqHz, magDbA);

				// 对 gaindB 的导数恒为 1
				paramsGradOut[0] += psi;

				// 对零点参数的导数
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * i;
					const double zm = paramsIn[base + 0];
					const double zt = paramsIn[base + 1];

					const double delta = w - zt;
					const double cosDelta = std::cos(delta);
					const double sinDelta = std::sin(delta);
					const double d2 = 1.0 + zm * zm - 2.0 * zm * cosDelta + kTiny;

					const double dMagDb_dZm = kDbScale * (zm - cosDelta) / d2;
					const double dMagDb_dZt = -kDbScale * zm * sinDelta / d2;

					paramsGradOut[base + 0] += psi * dMagDb_dZm;
					paramsGradOut[base + 1] += psi * dMagDb_dZt;
				}

				// 对极点参数的导数
				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * numOrders + 2 * i;
					const double pm = paramsIn[base + 0];
					const double pt = paramsIn[base + 1];

					const double delta = w - pt;
					const double cosDelta = std::cos(delta);
					const double sinDelta = std::sin(delta);
					const double d2 = 1.0 + pm * pm - 2.0 * pm * cosDelta + kTiny;

					const double dMagDb_dPm = -kDbScale * (pm - cosDelta) / d2;
					const double dMagDb_dPt = kDbScale * pm * sinDelta / d2;

					paramsGradOut[base + 0] += psi * dMagDb_dPm;
					paramsGradOut[base + 1] += psi * dMagDb_dPt;
				}
			}
		}

		double CalcTotalLoss(const std::vector<double>& paramsIn,
			MagErrorBase& magErrMethod,
			double sampleRate = 48000.0)
		{
			const int expectedNumParams = 1 + 4 * numOrders;
			//////assert((int)paramsIn.size() == expectedNumParams);

			const std::vector<double>& freqHzs = magErrMethod.GetFreqSpace();
			const std::vector<double>& targetDB = magErrMethod.GetGlobalDB();

			const int numPoints = (int)freqHzs.size();
			//////assert((int)targetDB.size() == numPoints);

			const double gainDb = paramsIn[0];
			double totalLoss = 0.0;

			for (int k = 0; k < numPoints; ++k)
			{
				const double freqHz = freqHzs[k];
				const double w = 2.0 * kPi * freqHz / sampleRate;

				double magDbA = gainDb;

				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * i;
					const double zm = paramsIn[base + 0];
					const double zt = paramsIn[base + 1];

					const double delta = w - zt;
					const double d2 = 1.0 + zm * zm - 2.0 * zm * std::cos(delta) + kTiny;

					magDbA += 10.0 * std::log10(d2);
				}

				for (int i = 0; i < numOrders; ++i)
				{
					const int base = 1 + 2 * numOrders + 2 * i;
					const double pm = paramsIn[base + 0];
					const double pt = paramsIn[base + 1];

					const double delta = w - pt;
					const double d2 = 1.0 + pm * pm - 2.0 * pm * std::cos(delta) + kTiny;

					magDbA -= 10.0 * std::log10(d2);
				}

				totalLoss += magErrMethod.Error(k, freqHz, magDbA);
			}

			return totalLoss;
		}

		double GetMagResp(const std::vector<double>& params, double freqHz, double sampleRate = 48000.0)
		{
			const int expectedNumParams = 1 + 4 * numOrders;
			//////assert((int)params.size() == expectedNumParams);

			const double gainDb = params[0];
			const double w = 2.0 * kPi * freqHz / sampleRate;

			double logMagDb = gainDb;

			// zeros
			for (int i = 0; i < numOrders; ++i)
			{
				const int base = 1 + 2 * i;
				const double zm = params[base + 0];
				const double zt = params[base + 1];

				const double delta = w - zt;
				const double d2 = 1.0 + zm * zm - 2.0 * zm * std::cos(delta) + kTiny;

				logMagDb += 10.0 * std::log10(d2);
			}

			// poles
			for (int i = 0; i < numOrders; ++i)
			{
				const int base = 1 + 2 * numOrders + 2 * i;
				const double pm = params[base + 0];
				const double pt = params[base + 1];

				const double delta = w - pt;
				const double d2 = 1.0 + pm * pm - 2.0 * pm * std::cos(delta) + kTiny;

				logMagDb -= 10.0 * std::log10(d2);
			}

			// dB -> linear
			const double magLinear = std::pow(10.0, logMagDb / 20.0);
			return magLinear;
		}
	};
	class AdamOptimizer // 纯数值优化器，不用关心参数具体的意义
	{
	private:
		double bestloss = 1e200;
		double learningRate = 1e-3;

		std::vector<double> bestParams;
		std::vector<double> nowParams;

		std::vector<double> m;
		std::vector<double> v;

		long long stepCount = 0;

	public:
		const std::vector<double> GetNowParams() const { return nowParams; }
		const std::vector<double> GetBestParams() const { return bestParams; }
		double GetBestLoss() const { return bestloss; }

		void SetupOptimizer(int numParams, std::vector<double> basin, double lr)
		{
			learningRate = lr;
			bestloss = 1e200;
			stepCount = 0;

			nowParams = basin;
			bestParams = basin;

			m.assign(numParams, 0.0);
			v.assign(numParams, 0.0);
		}

		void RunOptimizer(
			int numCycles,
			std::function<double/*total loss*/(std::vector<double>&/*params in*/, std::vector<double>&/*params grad out*/)> updateMethod)
		{
			const int numParams = (int)nowParams.size();

			// Adam常用默认值
			const double beta1 = 0.9;
			const double beta2 = 0.999;
			const double eps = 1e-8;

			// 简单全局梯度裁剪阈值，够保守
			const double gradClipNorm = 10.0;

			std::vector<double> grad(numParams, 0.0);

			for (int iter = 0; iter < numCycles; ++iter)
			{
				std::fill(grad.begin(), grad.end(), 0.0);

				double loss = updateMethod(nowParams, grad);

				if (std::isfinite(loss) && loss < bestloss)
				{
					bestloss = loss;
					bestParams = nowParams;
				}

				// 梯度有限性处理
				for (int i = 0; i < numParams; ++i)
				{
					if (!std::isfinite(grad[i]))
						grad[i] = 0.0;
				}

				// 全局L2 norm裁剪，防止某轮梯度过大直接炸掉
				double gradNorm2 = 0.0;
				for (int i = 0; i < numParams; ++i)
					gradNorm2 += grad[i] * grad[i];

				if (gradNorm2 > gradClipNorm * gradClipNorm)
				{
					const double gradNorm = std::sqrt(gradNorm2);
					const double scale = gradClipNorm / (gradNorm + 1e-30);
					for (int i = 0; i < numParams; ++i)
						grad[i] *= scale;
				}

				++stepCount;

				const double biasCorr1 = 1.0 - std::pow(beta1, (double)stepCount);
				const double biasCorr2 = 1.0 - std::pow(beta2, (double)stepCount);

				for (int i = 0; i < numParams; ++i)
				{
					const double g = grad[i];

					m[i] = beta1 * m[i] + (1.0 - beta1) * g;
					v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

					const double mHat = m[i] / biasCorr1;
					const double vHat = v[i] / biasCorr2;

					nowParams[i] -= learningRate * mHat / (std::sqrt(vHat) + eps);
				}
			}
		}
	};

	class LbfgsOptimizerLightweight
	{
	private:
		double bestloss = 1e200;
		double initStep = 1.0;
		int historySize = 10;

		std::vector<double> bestParams;
		std::vector<double> nowParams;
		std::vector<double> nowGrad;

		std::function<double(std::vector<double>&, std::vector<double>&)> updateMethod;

		struct HistPair
		{
			std::vector<double> s; // x_{k+1} - x_k
			std::vector<double> y; // g_{k+1} - g_k
			double rho = 0.0;      // 1 / (y^T s)
		};

		std::vector<HistPair> hist;

	private:
		static double Dot(const std::vector<double>& a, const std::vector<double>& b)
		{
			const int n = (int)a.size();
			double v = 0.0;
			for (int i = 0; i < n; ++i) v += a[i] * b[i];
			return v;
		}

		static double Norm2(const std::vector<double>& a)
		{
			return std::sqrt(std::max(0.0, Dot(a, a)));
		}

		static void AddScaled(std::vector<double>& dst, const std::vector<double>& src, double k)
		{
			const int n = (int)dst.size();
			for (int i = 0; i < n; ++i) dst[i] += src[i] * k;
		}

		static std::vector<double> Sub(const std::vector<double>& a, const std::vector<double>& b)
		{
			const int n = (int)a.size();
			std::vector<double> out(n);
			for (int i = 0; i < n; ++i) out[i] = a[i] - b[i];
			return out;
		}

		static void Scale(std::vector<double>& a, double k)
		{
			for (double& v : a) v *= k;
		}

		void PushHistory(const std::vector<double>& s, const std::vector<double>& y)
		{
			const double ys = Dot(y, s);
			if (!(ys > 1.0e-20))
				return;

			HistPair hp;
			hp.s = s;
			hp.y = y;
			hp.rho = 1.0 / ys;

			if ((int)hist.size() >= historySize)
				hist.erase(hist.begin());
			hist.push_back(std::move(hp));
		}

		void ComputeSearchDirection(const std::vector<double>& grad, std::vector<double>& dir) const
		{
			const int n = (int)grad.size();
			dir = grad;
			for (double& v : dir) v = -v;

			if (hist.empty())
				return;

			const int m = (int)hist.size();
			std::vector<double> q = grad;
			std::vector<double> alpha(m, 0.0);

			for (int i = m - 1; i >= 0; --i)
			{
				alpha[i] = hist[i].rho * Dot(hist[i].s, q);
				AddScaled(q, hist[i].y, -alpha[i]);
			}

			double gamma = 1.0;
			{
				const HistPair& last = hist.back();
				const double yy = Dot(last.y, last.y);
				const double ys = Dot(last.y, last.s);
				if (yy > 1.0e-20)
					gamma = ys / yy;
			}

			std::vector<double> r = q;
			Scale(r, gamma);

			for (int i = 0; i < m; ++i)
			{
				const double beta = hist[i].rho * Dot(hist[i].y, r);
				AddScaled(r, hist[i].s, alpha[i] - beta);
			}

			dir = r;
			for (double& v : dir) v = -v;
		}

	public:
		void SetupOptimizer(int numParams, std::vector<double> basin, double stepScale = 1.0)
		{
			////assert(numParams > 0);
			////assert((int)basin.size() == numParams);

			bestloss = 1e200;
			initStep = stepScale;

			nowParams = basin;
			bestParams = basin;
			nowGrad.assign(numParams, 0.0);
			hist.clear();
		}

		void SetBasin(const std::vector<double>& basin)
		{
			nowParams = basin;
			bestParams = basin;
			nowGrad.assign(basin.size(), 0.0);
			hist.clear();
			bestloss = 1e200;
		}

		void SetErrorFunc(std::function<double(std::vector<double>&, std::vector<double>&)> fn)
		{
			updateMethod = std::move(fn);
		}

		void GetNowVec(std::vector<float>& out) const
		{
			out.resize(nowParams.size());
			for (size_t i = 0; i < nowParams.size(); ++i) out[i] = (float)nowParams[i];
		}

		void GetBestVec(std::vector<float>& out) const
		{
			out.resize(bestParams.size());
			for (size_t i = 0; i < bestParams.size(); ++i) out[i] = (float)bestParams[i];
		}

		void GetNowVec(std::vector<double>& out) const
		{
			out = nowParams;
		}

		void GetBestVec(std::vector<double>& out) const
		{
			out = bestParams;
		}

		double GetBestLoss() const { return bestloss; }

		void RunOptimizer(int numCycles)
		{
			////assert(updateMethod);
			////assert(!nowParams.empty());
			if (numCycles <= 0) return;

			const int n = (int)nowParams.size();
			std::vector<double> grad(n, 0.0);
			double loss = updateMethod(nowParams, grad);

			for (int i = 0; i < n; ++i)
				if (!std::isfinite(grad[i])) grad[i] = 0.0;

			nowGrad = grad;

			if (std::isfinite(loss) && loss < bestloss)
			{
				bestloss = loss;
				bestParams = nowParams;
			}

			for (int iter = 0; iter < numCycles; ++iter)
			{
				std::vector<double> dir;
				ComputeSearchDirection(nowGrad, dir);

				double gtd = Dot(nowGrad, dir);

				// 如果不是下降方向，退回最速下降
				if (!(gtd < -1.0e-20))
				{
					dir = nowGrad;
					for (double& v : dir) v = -v;
					gtd = Dot(nowGrad, dir);
				}

				const double gradNorm = Norm2(nowGrad);
				if (!(gradNorm > 1.0e-14))
					break;

				// 简单回溯 Armijo
				double step = initStep;
				const double c1 = 1.0e-4;
				const double shrink = 0.5;
				const int maxLineSearch = 20;

				std::vector<double> trialParams(n), trialGrad(n, 0.0);
				double trialLoss = 1e300;
				bool accepted = false;

				for (int ls = 0; ls < maxLineSearch; ++ls)
				{
					trialParams = nowParams;
					AddScaled(trialParams, dir, step);

					std::fill(trialGrad.begin(), trialGrad.end(), 0.0);
					trialLoss = updateMethod(trialParams, trialGrad);

					for (int i = 0; i < n; ++i)
						if (!std::isfinite(trialGrad[i])) trialGrad[i] = 0.0;

					if (std::isfinite(trialLoss) && trialLoss <= loss + c1 * step * gtd)
					{
						accepted = true;
						break;
					}

					step *= shrink;
				}

				if (!accepted)
				{
					// 线搜索失败时，退化成很小一步的梯度下降
					step = 1.0e-3;
					trialParams = nowParams;
					AddScaled(trialParams, dir, step);

					std::fill(trialGrad.begin(), trialGrad.end(), 0.0);
					trialLoss = updateMethod(trialParams, trialGrad);

					for (int i = 0; i < n; ++i)
						if (!std::isfinite(trialGrad[i])) trialGrad[i] = 0.0;

					if (!std::isfinite(trialLoss))
						break;
				}

				std::vector<double> s = Sub(trialParams, nowParams);
				std::vector<double> y = Sub(trialGrad, nowGrad);

				PushHistory(s, y);

				nowParams = std::move(trialParams);
				nowGrad = std::move(trialGrad);
				loss = trialLoss;

				if (std::isfinite(loss) && loss < bestloss)
				{
					bestloss = loss;
					bestParams = nowParams;
				}

				// 对下一轮初始步长做一个轻微自适应
				if (step < initStep * 0.25)
					initStep = std::max(1.0e-3, initStep * 0.7);
				else if (step > initStep * 0.9)
					initStep = std::min(2.0, initStep * 1.05);
			}
		}
	};



	class MatchedComplexIIRDesignAnalytic : public IIRDesignBase
	{
	private:
		static constexpr float M_PI_F = 3.14159265358979323846f;
		static constexpr int numPoints = 100;
		static constexpr int numOrders = 2;
		static constexpr int numParams = 1 + 4 * numOrders; // 9

		AdamOptimizer optAdam;
		LbfgsOptimizerLightweight optLbfgs;
		AnalogPrototypeFilter prototype;
		ComplexIIRGradient<numOrders> gradCalc;

		struct DefaultMagError : public MagErrorBase
		{
			double Error(int i, double freqhz, double iirmagdb) override
			{
				const double errDB = iirmagdb - globalDB[i];
				const double e2 = errDB * errDB;
				const double e1 = std::abs(errDB);

				// 这里延续你原来的风格
				const double ev = e1 * 0.1 + e2 * 0.9;
				const double freqk = freqhz / 48000.0 * 0.2 + 0.8;

				// 如果以后你想继续折腾误差，这里直接改就行
				return ev * freqk;
			}
		};

		DefaultMagError magErr;

		std::vector<float> coeffs;
		std::vector<double> coeffsD;
		std::vector<double> targetMagLinD;
		std::vector<double> freqSpaceD;

		float freqSpace[numPoints] = { 0.0f };
		float magdBSpace[numPoints] = { 0.0f };
		float magLinSpace[numPoints] = { 0.0f };

		float meanTargetDB = 0.0f;

		AnalogFilterType now_type = AnalogFilterType::LP;
		float now_fc = 5000.0f;
		float now_q = 0.707f;
		float now_gain = 10.0f;
		float now_stages = 2.0f;

		int totalCycles = 0;

		bool usingLbfgs = false;
	private:
		float randNormV()
		{
			const float a = (float)std::rand() / (float)RAND_MAX;
			const float b = (float)std::rand() / (float)RAND_MAX;
			return a * b * (std::rand() % 2 ? 1.0f : -1.0f);
		}

		float spaceTransitionV = 0.75f;
		float TransitionSpace(float minv, float maxv, float normx) const
		{
			const float logspace = std::exp(std::log(minv) + (std::log(maxv) - std::log(minv)) * normx);
			const float linspace = minv + (maxv - minv) * normx;
			return linspace * (1.0f - spaceTransitionV) + logspace * spaceTransitionV;
		}

		void FloatToDouble(const std::vector<float>& in, std::vector<double>& out) const
		{
			out.resize(in.size());
			for (size_t i = 0; i < in.size(); ++i) out[i] = (double)in[i];
		}

		void DoubleToFloat(const std::vector<double>& in, std::vector<float>& out) const
		{
			out.resize(in.size());
			for (size_t i = 0; i < in.size(); ++i) out[i] = (float)in[i];
		}

		void SetupInitialParams()
		{
			coeffs.resize(numParams);

			// gainDb
			coeffs[0] = 0.0f;

			// zeros: 初始化在单位圆内侧附近的小范围
			coeffs[1] = 0.90f * randNormV();
			coeffs[2] = 0.90f * randNormV();
			coeffs[3] = 0.90f * randNormV();
			coeffs[4] = 0.90f * randNormV();

			// poles: 更保守一点，先放小一些，避免一上来爆梯度
			coeffs[5] = 0.90f * randNormV();
			coeffs[6] = 0.90f * randNormV();
			coeffs[7] = 0.90f * randNormV();
			coeffs[8] = 0.90f * randNormV();

			FloatToDouble(coeffs, coeffsD);
		}

		double LossAndGrad(std::vector<double>& params, std::vector<double>& grad)
		{
			gradCalc.CalcGradient(params, grad, magErr, 48000.0);
			return gradCalc.CalcTotalLoss(params, magErr, 48000.0);
		}

	public:
		MatchedComplexIIRDesignAnalytic(int iirtype = 0)
		{
			coeffs.resize(numParams);
			coeffsD.resize(numParams);
			targetMagLinD.resize(numPoints);
			freqSpaceD.resize(numPoints);
			Init();
		}

		void Init() override
		{
			std::srand(31415926);

			for (int i = 0; i < numPoints; ++i)
			{
				freqSpace[i] = TransitionSpace(20.0f, 24000.0f, (float)i / (float)(numPoints - 1));
				freqSpaceD[i] = (double)freqSpace[i];
			}

			SetupInitialParams();

			//optAdam.SetupOptimizer(numParams, coeffsD, 0.3); // 初值可再调
			//optLbfgs.SetupOptimizer(numParams, coeffsD, 0.5);
			totalCycles = 0;
		}

		void SetupAnalogPrototype(AnalogFilterType type, float fc, float q, float gain, float stages) override
		{
			Init();

			now_type = type;
			now_fc = fc;
			now_q = q;
			now_gain = gain;
			now_stages = stages;

			meanTargetDB = 0.0f;

			for (int i = 0; i < numPoints; ++i)
			{
				const float freqhz = freqSpace[i];
				const float mag = prototype.GetMapResp(type, freqhz, fc, q, gain, stages);

				magLinSpace[i] = std::max(mag, 1.0e-30f);
				magdBSpace[i] = 20.0f * std::log10(std::max(mag, 1.0e-30f));
				meanTargetDB += magdBSpace[i];

				targetMagLinD[i] = (double)magLinSpace[i];
			}
			meanTargetDB /= (float)numPoints;

			magErr.SetupMagLinearGlobal(targetMagLinD, freqSpaceD, numPoints);

			// 重置优化器 basin
			SetupInitialParams();
			optAdam.SetupOptimizer(numParams, coeffsD, 0.1);
			optLbfgs.SetupOptimizer(numParams, coeffsD, 0.9);
			totalCycles = 0;
			usingLbfgs = false;
		}

		void RunOptimizer(int numCycles, int maxCycles) override
		{
			if (!usingLbfgs)
			{
				optAdam.RunOptimizer(numCycles,
					[this](std::vector<double>& params, std::vector<double>& grad) -> double
					{
						return LossAndGrad(params, grad);
					});
			}
			else
			{
				optLbfgs.RunOptimizer(numCycles);
			}

			totalCycles += numCycles;

			if (totalCycles > maxCycles * 10.0 / 100.0 && !usingLbfgs)
			{
				usingLbfgs = true;
				std::vector<double> bestAdam = optAdam.GetBestParams();
				optLbfgs.SetBasin(bestAdam);
				optLbfgs.SetErrorFunc(
					[this](std::vector<double>& params, std::vector<double>& grad) -> double
					{
						return LossAndGrad(params, grad);
					});
			}
		}

		void RunOptimizerDirect(int adamCycles = 40, int lbfgsCycles = 160) override
		{
			optAdam.RunOptimizer(adamCycles,
				[this](std::vector<double>& params, std::vector<double>& grad) -> double
				{
					return LossAndGrad(params, grad);
				});

			std::vector<double> bestAdam = optAdam.GetBestParams();
			optLbfgs.SetBasin(bestAdam);
			optLbfgs.SetErrorFunc(
				[this](std::vector<double>& params, std::vector<double>& grad) -> double
				{
					return LossAndGrad(params, grad);
				});
			optLbfgs.RunOptimizer(lbfgsCycles);
			usingLbfgs = true;

			totalCycles += adamCycles + lbfgsCycles;
		}

		void GetNowCoeffs(std::vector<float>& outCoeffs) override
		{
			if (usingLbfgs)
			{
				std::vector<double> now;
				optLbfgs.GetNowVec(now);
				DoubleToFloat(now, outCoeffs);
			}
			else
			{
				const std::vector<double> now = optAdam.GetNowParams();
				DoubleToFloat(now, outCoeffs);
			}
		}
		void GetBestCoeffs(std::vector<float>& outCoeffs) override
		{
			if (usingLbfgs)
			{
				std::vector<double> best;
				optLbfgs.GetBestVec(best);
				DoubleToFloat(best, outCoeffs);
			}
			else
			{
				const std::vector<double> best = optAdam.GetBestParams();
				DoubleToFloat(best, outCoeffs);
			}
		}

		float GetPrototypeResp(float freqhz) override
		{
			return prototype.GetMapResp(now_type, freqhz, now_fc, now_q, now_gain, now_stages);
		}

		float GetNowIIRResp(float freqhz) override
		{
			std::vector<double> now;
			if (usingLbfgs) optLbfgs.GetNowVec(now);
			else now = optAdam.GetNowParams();
			return (float)gradCalc.GetMagResp(now, (double)freqhz, 48000.0);
		}
		float GetBestIIRResp(float freqhz) override
		{
			std::vector<double> best;
			if (usingLbfgs) optLbfgs.GetBestVec(best);
			else best = optAdam.GetBestParams();
			return (float)gradCalc.GetMagResp(best, (double)freqhz, 48000.0);
		}
	};
}