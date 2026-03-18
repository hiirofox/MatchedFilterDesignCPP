#pragma once

#include <vector>
#include <functional>

class OptimizerBase
{
public:
	using ErrorFuncType = std::function<float(std::vector<float>&)>;

	virtual ~OptimizerBase() = default;

	virtual void SetErrorFunc(ErrorFuncType errorfunc) = 0;
	virtual void SetupOptimizer(int numPoints, std::vector<float> initVec, float learningRate) = 0;
	virtual void SetBasin(std::vector<float>& best) = 0;
	virtual void RunOptimizer(int numCycles) = 0;

	virtual int GetBestVec(std::vector<float>& best) = 0;
	virtual int GetNowVec(std::vector<float>& nowv) = 0;
	virtual float GetNowError() = 0;
};

class Optimizer :public OptimizerBase
{
private:
	using ErrorFuncType = std::function<float(std::vector<float>&)>;
	ErrorFuncType errfunc = [](std::vector<float>&) { return 0; };

	int numPoints = 0;
	std::vector<float> bestVec;
	std::vector<float> nowVec;
	std::vector<float> nowVecCopy;
	std::vector<float> gradVec;
	std::vector<float> momentVec;
	float lr = 0.1, momentApply = 0;
	float err = 1e20, besterr = 1e20;
	int cycles = 0, bestCycleAt = 0;

public:
	Optimizer()
	{

	}
	void SetErrorFunc(ErrorFuncType errorfunc) final override { this->errfunc = errorfunc; }
	void SetupOptimizer(int numPoints, std::vector<float> initVec, float learningRate)final override
	{
		bestVec.resize(numPoints, 0);
		nowVec.resize(numPoints, 0);
		nowVecCopy.resize(numPoints, 0);
		gradVec.resize(numPoints, 0);
		momentVec.resize(numPoints, 0);

		this->numPoints = numPoints;
		nowVec = initVec;
		lr = learningRate;
		momentApply = 0.1;//先试试无动量
		besterr = 1e20;
		cycles = 0;
	}
	void SetBasin(std::vector<float>& best)final override
	{
		nowVec = best;
		bestVec = best;
	}
	void RunOptimizer(int numCycles)final override
	{
		for (int i = 0; i < numCycles; ++i, ++cycles)
		{
			err = errfunc(nowVec);
			if (err < besterr)
			{
				besterr = err;
				bestVec = nowVec;
				bestCycleAt = cycles;
			}
			for (int j = 0; j < numPoints; ++j)
			{
				float h = 1e-4f * std::max(1.0f, std::abs(nowVec[j]));

				nowVecCopy = nowVec;
				nowVecCopy[j] += h;
				float errj = errfunc(nowVecCopy);
				float gradj = (errj - err) / h;
				gradVec[j] = gradj;
				momentVec[j] = momentVec[j] * momentApply + gradj;
			}
			for (int j = 0; j < numPoints; ++j)
			{
				float veloj = momentVec[j];//梯度+动量的一点点应用
				nowVec[j] -= veloj * lr;//向斜率反方向前进，到达err较小的谷
			}
		}
	}
	int GetBestVec(std::vector<float>& best)final override
	{
		best = bestVec;
		return bestCycleAt;
	}
	int GetNowVec(std::vector<float>& nowv)final override
	{
		nowv = nowVec;
		return cycles;
	}
	float GetNowError()final override
	{
		return err;
	}
};

class LbfgsOptimizer :public OptimizerBase
{
private:
	using ErrorFuncType = std::function<float(std::vector<float>&)>;
	ErrorFuncType errfunc = [](std::vector<float>&) { return 0.0f; };

	int numPoints = 0;

	std::vector<float> bestVec;
	std::vector<float> nowVec;
	std::vector<float> nowVecCopy;
	std::vector<float> gradVec;

	float lr = 1.0f;
	float err = 1e20f;
	float besterr = 1e20f;
	int cycles = 0;
	int bestCycleAt = 0;

	struct HistoryPair
	{
		std::vector<float> s; // x_{k+1} - x_k
		std::vector<float> y; // g_{k+1} - g_k
		float ys = 0.0f;      // dot(y, s)
	};

	std::vector<HistoryPair> history;
	int historySize = 8;

private:
	static float Dot(const std::vector<float>& a, const std::vector<float>& b)
	{
		float v = 0.0f;
		const int n = (int)a.size();
		for (int i = 0; i < n; ++i) v += a[i] * b[i];
		return v;
	}

	static float Norm2(const std::vector<float>& a)
	{
		return sqrtf(std::max(0.0f, Dot(a, a)));
	}

	static void AddScaled(std::vector<float>& dst, const std::vector<float>& src, float scale)
	{
		const int n = (int)dst.size();
		for (int i = 0; i < n; ++i) dst[i] += src[i] * scale;
	}

	static std::vector<float> Sub(const std::vector<float>& a, const std::vector<float>& b)
	{
		const int n = (int)a.size();
		std::vector<float> out(n);
		for (int i = 0; i < n; ++i) out[i] = a[i] - b[i];
		return out;
	}

	float EvalError(std::vector<float>& x)
	{
		return errfunc(x);
	}

	void EvalGradient(const std::vector<float>& x, float fx, std::vector<float>& g)
	{
		g.resize(numPoints);
		nowVecCopy = x;

		for (int j = 0; j < numPoints; ++j)
		{
			const float base = x[j];
			const float h = 1e-4f * std::max(1.0f, std::abs(base));

			nowVecCopy[j] = base + h;
			float f1 = EvalError(nowVecCopy);

			nowVecCopy[j] = base - h;
			float f2 = EvalError(nowVecCopy);

			g[j] = (f1 - f2) / (2.0f * h);
			nowVecCopy[j] = base;
		}
	}

	void ComputeLbfgsDirection(const std::vector<float>& grad, std::vector<float>& dir)
	{
		const int m = (int)history.size();
		dir = grad;
		if (m == 0)
		{
			for (float& v : dir) v = -v;
			return;
		}

		std::vector<float> alpha(m, 0.0f);
		std::vector<float> rho(m, 0.0f);

		for (int i = 0; i < m; ++i)
			rho[i] = (history[i].ys != 0.0f) ? (1.0f / history[i].ys) : 0.0f;

		for (int i = m - 1; i >= 0; --i)
		{
			alpha[i] = rho[i] * Dot(history[i].s, dir);
			AddScaled(dir, history[i].y, -alpha[i]);
		}

		float gamma = 1.0f;
		{
			const HistoryPair& last = history.back();
			const float yy = Dot(last.y, last.y);
			if (yy > 1e-20f) gamma = last.ys / yy;
		}

		for (float& v : dir) v *= gamma;

		for (int i = 0; i < m; ++i)
		{
			const float beta = rho[i] * Dot(history[i].y, dir);
			AddScaled(dir, history[i].s, alpha[i] - beta);
		}

		for (float& v : dir) v = -v;
	}

	float LineSearch(
		const std::vector<float>& x,
		float fx,
		const std::vector<float>& grad,
		const std::vector<float>& dir,
		std::vector<float>& xNew,
		float& fNew)
	{
		float step = lr;
		const float c1 = 1e-4f;
		const float dg = Dot(grad, dir);

		xNew = x;

		if (dg >= 0.0f)
		{
			step = 1e-3f;
			for (int i = 0; i < numPoints; ++i) xNew[i] = x[i] - grad[i] * step;
			fNew = EvalError(xNew);
			return step;
		}

		for (int iter = 0; iter < 20; ++iter)
		{
			for (int i = 0; i < numPoints; ++i)
				xNew[i] = x[i] + dir[i] * step;

			fNew = EvalError(xNew);

			if (fNew <= fx + c1 * step * dg)
				return step;

			step *= 0.5f;
		}

		for (int i = 0; i < numPoints; ++i)
			xNew[i] = x[i];

		fNew = fx;
		return 0.0f;
	}

public:
	LbfgsOptimizer() = default;

	void SetErrorFunc(ErrorFuncType errorfunc)final override
	{
		this->errfunc = errorfunc;
	}

	void SetupOptimizer(int numPoints, std::vector<float> initVec, float learningRate)final override
	{
		this->numPoints = numPoints;
		this->lr = learningRate;

		nowVec = initVec;
		nowVec.resize(numPoints, 0.0f);

		bestVec = nowVec;
		nowVecCopy.resize(numPoints, 0.0f);
		gradVec.resize(numPoints, 0.0f);

		history.clear();

		err = 1e20f;
		besterr = 1e20f;
		cycles = 0;
		bestCycleAt = 0;
	}

	void SetBasin(std::vector<float>& best)final override
	{
		nowVec = best;
		bestVec = best;
	}

	void RunOptimizer(int numCycles)final override
	{
		if (numPoints <= 0) return;

		std::vector<float> xNew(numPoints);
		std::vector<float> gNew(numPoints);
		std::vector<float> dir(numPoints);

		for (int i = 0; i < numCycles; ++i, ++cycles)
		{
			err = EvalError(nowVec);

			if (err < besterr)
			{
				besterr = err;
				bestVec = nowVec;
				bestCycleAt = cycles;
			}

			EvalGradient(nowVec, err, gradVec);

			const float gnorm = Norm2(gradVec);
			if (gnorm < 1e-7f)
				return;

			ComputeLbfgsDirection(gradVec, dir);

			float fNew = err;
			const float step = LineSearch(nowVec, err, gradVec, dir, xNew, fNew);

			if (step <= 0.0f)
				return;

			EvalGradient(xNew, fNew, gNew);

			std::vector<float> s = Sub(xNew, nowVec);
			std::vector<float> y = Sub(gNew, gradVec);

			const float ys = Dot(y, s);
			if (ys > 1e-12f)
			{
				if ((int)history.size() >= historySize)
					history.erase(history.begin());

				history.push_back({ std::move(s), std::move(y), ys });
			}

			nowVec = xNew;
			gradVec = gNew;
			err = fNew;

			if (err < besterr)
			{
				besterr = err;
				bestVec = nowVec;
				bestCycleAt = cycles;
			}
		}
	}

	int GetBestVec(std::vector<float>& best)final override
	{
		best = bestVec;
		return bestCycleAt;
	}

	int GetNowVec(std::vector<float>& nowv)final override
	{
		nowv = nowVec;
		return cycles;
	}

	float GetNowError()final override
	{
		return err;
	}
};

class LbfgsOptimizer2 : public OptimizerBase
{
private:
	using ErrorFuncType = std::function<float(std::vector<float>&)>;
	ErrorFuncType errfunc = [](std::vector<float>&) { return 0.0f; };

	int numPoints = 0;

	std::vector<float> bestVec;
	std::vector<float> nowVec;
	std::vector<float> nowVecCopy;
	std::vector<float> gradVec;

	double lr = 1.0;
	double err = 1e300;
	double besterr = 1e300;
	int cycles = 0;
	int bestCycleAt = 0;

	int stallCount = 0;
	int lineSearchFailCount = 0;

	int historySize = 8;

	double gradTol = 1e-6;
	double relImproveTol = 1e-7;
	int stallLimit = 24;
	int lineSearchFailLimit = 8;
	double minStep = 1e-10;
	double maxStep = 10.0;
	double fdRelStep = 1e-3;

	struct HistoryPair
	{
		std::vector<double> s;
		std::vector<double> y;
		double ys = 0.0;
	};

	std::vector<HistoryPair> history;

private:
	static double Dot(const std::vector<double>& a, const std::vector<double>& b)
	{
		double v = 0.0;
		const int n = (int)a.size();
		for (int i = 0; i < n; ++i)
			v += a[i] * b[i];
		return v;
	}

	static double Norm2(const std::vector<double>& a)
	{
		return sqrt(std::max(0.0, Dot(a, a)));
	}

	static void AddScaled(std::vector<double>& dst, const std::vector<double>& src, double scale)
	{
		const int n = (int)dst.size();
		for (int i = 0; i < n; ++i)
			dst[i] += src[i] * scale;
	}

	static std::vector<double> Sub(const std::vector<double>& a, const std::vector<double>& b)
	{
		const int n = (int)a.size();
		std::vector<double> out(n);
		for (int i = 0; i < n; ++i)
			out[i] = a[i] - b[i];
		return out;
	}

	static void CopyFloatToDouble(const std::vector<float>& src, std::vector<double>& dst)
	{
		const int n = (int)src.size();
		dst.resize(n);
		for (int i = 0; i < n; ++i)
			dst[i] = (double)src[i];
	}

	static void CopyDoubleToFloat(const std::vector<double>& src, std::vector<float>& dst)
	{
		const int n = (int)src.size();
		dst.resize(n);
		for (int i = 0; i < n; ++i)
			dst[i] = (float)src[i];
	}

	double EvalError(const std::vector<double>& x)
	{
		nowVecCopy.resize(numPoints);
		for (int i = 0; i < numPoints; ++i)
			nowVecCopy[i] = (float)x[i];
		return (double)errfunc(nowVecCopy);
	}

	void EvalGradient(const std::vector<double>& x, double fx, std::vector<double>& g)
	{
		(void)fx;
		g.resize(numPoints);

		std::vector<double> xt = x;

		for (int j = 0; j < numPoints; ++j)
		{
			const double base = x[j];
			const double h = fdRelStep * (1.0 + std::abs(base));

			xt[j] = base + h;
			const double f1 = EvalError(xt);

			xt[j] = base - h;
			const double f2 = EvalError(xt);

			g[j] = (f1 - f2) / (2.0 * h);
			xt[j] = base;
		}
	}

	void ComputeLbfgsDirection(const std::vector<double>& grad, std::vector<double>& dir)
	{
		const int m = (int)history.size();
		dir = grad;

		if (m == 0)
		{
			for (double& v : dir)
				v = -v;
			return;
		}

		std::vector<double> alpha(m, 0.0);
		std::vector<double> rho(m, 0.0);

		for (int i = 0; i < m; ++i)
			rho[i] = (std::abs(history[i].ys) > 1e-30) ? (1.0 / history[i].ys) : 0.0;

		for (int i = m - 1; i >= 0; --i)
		{
			alpha[i] = rho[i] * Dot(history[i].s, dir);
			AddScaled(dir, history[i].y, -alpha[i]);
		}

		double gamma = 1.0;
		{
			const HistoryPair& last = history.back();
			const double yy = Dot(last.y, last.y);
			if (yy > 1e-30)
				gamma = last.ys / yy;
		}

		for (double& v : dir)
			v *= gamma;

		for (int i = 0; i < m; ++i)
		{
			const double beta = rho[i] * Dot(history[i].y, dir);
			AddScaled(dir, history[i].s, alpha[i] - beta);
		}

		for (double& v : dir)
			v = -v;
	}

	bool ArmijoBacktracking(
		const std::vector<double>& x,
		double fx,
		const std::vector<double>& grad,
		const std::vector<double>& dir,
		std::vector<double>& xNew,
		double& fNew,
		double& usedStep)
	{
		const double c1 = 1e-4;
		const double dg0 = Dot(grad, dir);

		xNew = x;
		fNew = fx;
		usedStep = 0.0;

		if (!(dg0 < 0.0))
			return false;

		double step = std::min(std::max(lr, minStep), maxStep);

		for (int iter = 0; iter < 28; ++iter)
		{
			for (int i = 0; i < numPoints; ++i)
				xNew[i] = x[i] + step * dir[i];

			fNew = EvalError(xNew);

			if (fNew <= fx + c1 * step * dg0)
			{
				usedStep = step;
				return true;
			}

			step *= 0.5;
			if (step < minStep)
				break;
		}

		xNew = x;
		fNew = fx;
		usedStep = 0.0;
		return false;
	}

	bool TryStep(
		const std::vector<double>& x,
		double fx,
		const std::vector<double>& grad,
		std::vector<double>& dir,
		std::vector<double>& xNew,
		double& fNew,
		double& usedStep)
	{
		if (ArmijoBacktracking(x, fx, grad, dir, xNew, fNew, usedStep))
			return true;

		history.clear();

		dir = grad;
		for (double& v : dir)
			v = -v;

		if (ArmijoBacktracking(x, fx, grad, dir, xNew, fNew, usedStep))
			return true;

		return false;
	}

	void PushHistory(const std::vector<double>& s, const std::vector<double>& y)
	{
		const double ys = Dot(y, s);
		const double ss = Dot(s, s);
		const double yy = Dot(y, y);

		const double quality = 1e-8 * sqrt(std::max(1e-30, ss * yy));

		if (ys > quality)
		{
			if ((int)history.size() >= historySize)
				history.erase(history.begin());
			history.push_back({ s, y, ys });
		}
		else
		{
			history.clear();
		}
	}

public:
	LbfgsOptimizer2() = default;

	void SetErrorFunc(ErrorFuncType errorfunc) final override
	{
		this->errfunc = errorfunc;
	}

	void SetupOptimizer(int numPoints, std::vector<float> initVec, float learningRate) final override
	{
		this->numPoints = numPoints;
		this->lr = std::max(1e-12, (double)learningRate);

		nowVec = std::move(initVec);
		nowVec.resize(numPoints, 0.0f);

		bestVec = nowVec;
		nowVecCopy.resize(numPoints, 0.0f);
		gradVec.assign(numPoints, 0.0f);

		err = 1e300;
		besterr = 1e300;
		cycles = 0;
		bestCycleAt = 0;

		stallCount = 0;
		lineSearchFailCount = 0;

		history.clear();
	}
	void SetBasin(std::vector<float>& best)final override
	{
		nowVec = best;
		bestVec = best;
	}

	void RunOptimizer(int numCycles) final override
	{
		if (numPoints <= 0 || numCycles <= 0)
			return;

		std::vector<double> x;
		std::vector<double> g;
		std::vector<double> xNew;
		std::vector<double> gNew;
		std::vector<double> dir;

		CopyFloatToDouble(nowVec, x);
		xNew.resize(numPoints);
		g.resize(numPoints);
		gNew.resize(numPoints);
		dir.resize(numPoints);

		err = EvalError(x);

		if (err < besterr)
		{
			besterr = err;
			bestVec = nowVec;
			bestCycleAt = cycles;
		}

		EvalGradient(x, err, g);

		for (int iter = 0; iter < numCycles; ++iter, ++cycles)
		{
			const double gnorm = Norm2(g);

			if (gnorm < gradTol && stallCount >= stallLimit / 2)
				break;

			ComputeLbfgsDirection(g, dir);

			double fNew = err;
			double usedStep = 0.0;
			bool ok = TryStep(x, err, g, dir, xNew, fNew, usedStep);

			if (!ok)
			{
				++lineSearchFailCount;
				lr = std::max(minStep, lr * 0.5);

				if (lineSearchFailCount >= lineSearchFailLimit)
					break;

				++stallCount;
				continue;
			}

			lineSearchFailCount = 0;

			EvalGradient(xNew, fNew, gNew);

			std::vector<double> s = Sub(xNew, x);
			std::vector<double> y = Sub(gNew, g);
			PushHistory(s, y);

			x = xNew;
			g = gNew;
			err = fNew;

			CopyDoubleToFloat(x, nowVec);
			CopyDoubleToFloat(g, gradVec);

			if (err < besterr)
			{
				const double prevBest = besterr;
				besterr = err;
				bestVec = nowVec;
				bestCycleAt = cycles;

				const double denom = std::max(1.0, std::abs(prevBest));
				const double relImprove = (prevBest - err) / denom;

				if (relImprove > relImproveTol)
					stallCount = 0;
				else
					++stallCount;
			}
			else
			{
				++stallCount;
			}

			lr = std::min(maxStep, std::max(minStep, usedStep * 1.5));

			if (stallCount >= stallLimit)
				break;
		}

		nowVec = bestVec;
		err = besterr;
	}

	int GetBestVec(std::vector<float>& best) final override
	{
		best = bestVec;
		return bestCycleAt;
	}

	int GetNowVec(std::vector<float>& nowv) final override
	{
		nowv = nowVec;
		return cycles;
	}

	float GetNowError() final override
	{
		return (float)err;
	}
};

class LbfgsOptimizer3 : public OptimizerBase
{
private:
	using ErrorFuncType = std::function<float(std::vector<float>&)>;
	ErrorFuncType errfunc = [](std::vector<float>&) { return 0.0f; };

	int numPoints = 0;

	std::vector<float> bestVec;
	std::vector<float> nowVec;
	std::vector<float> nowVecCopy;
	std::vector<float> gradVec;

	double lr = 1.0;
	double err = 1e300;
	double besterr = 1e300;
	int cycles = 0;
	int bestCycleAt = 0;

	int stallCount = 0;
	int lineSearchFailCount = 0;
	int noImproveCount = 0;

	int historySize = 8;

	double gradTol = 1e-6;
	double relImproveTol = 1e-7;
	int stallLimit = 24;
	int lineSearchFailLimit = 8;
	int noImproveLimit = 64;

	double minStep = 1e-12;
	double tinyStepTol = 1e-12;
	double maxStep = 10.0;
	double fdRelStep = 1e-3;

	double refineStepGrow = 1.2;
	double refineStepShrink = 0.5;
	double basinInitStep = 1e-4;

	bool persistentMode = true;
	bool convergedHint = false;

	struct HistoryPair
	{
		std::vector<double> s;
		std::vector<double> y;
		double ys = 0.0;
	};

	std::vector<HistoryPair> history;

private:
	static double Dot(const std::vector<double>& a, const std::vector<double>& b)
	{
		double v = 0.0;
		const int n = (int)a.size();
		for (int i = 0; i < n; ++i)
			v += a[i] * b[i];
		return v;
	}

	static double Norm2(const std::vector<double>& a)
	{
		return sqrt(std::max(0.0, Dot(a, a)));
	}

	static void AddScaled(std::vector<double>& dst, const std::vector<double>& src, double scale)
	{
		const int n = (int)dst.size();
		for (int i = 0; i < n; ++i)
			dst[i] += src[i] * scale;
	}

	static std::vector<double> Sub(const std::vector<double>& a, const std::vector<double>& b)
	{
		const int n = (int)a.size();
		std::vector<double> out(n);
		for (int i = 0; i < n; ++i)
			out[i] = a[i] - b[i];
		return out;
	}

	static void CopyFloatToDouble(const std::vector<float>& src, std::vector<double>& dst)
	{
		const int n = (int)src.size();
		dst.resize(n);
		for (int i = 0; i < n; ++i)
			dst[i] = (double)src[i];
	}

	static void CopyDoubleToFloat(const std::vector<double>& src, std::vector<float>& dst)
	{
		const int n = (int)src.size();
		dst.resize(n);
		for (int i = 0; i < n; ++i)
			dst[i] = (float)src[i];
	}

	double EvalError(const std::vector<double>& x)
	{
		nowVecCopy.resize(numPoints);
		for (int i = 0; i < numPoints; ++i)
			nowVecCopy[i] = (float)x[i];
		return (double)errfunc(nowVecCopy);
	}

	void EvalGradient(const std::vector<double>& x, double fx, std::vector<double>& g)
	{
		(void)fx;
		g.resize(numPoints);

		std::vector<double> xt = x;

		for (int j = 0; j < numPoints; ++j)
		{
			const double base = x[j];
			const double h = fdRelStep * (1.0 + std::abs(base));

			xt[j] = base + h;
			const double f1 = EvalError(xt);

			xt[j] = base - h;
			const double f2 = EvalError(xt);

			g[j] = (f1 - f2) / (2.0 * h);
			xt[j] = base;
		}
	}

	void ComputeLbfgsDirection(const std::vector<double>& grad, std::vector<double>& dir)
	{
		const int m = (int)history.size();
		dir = grad;

		if (m == 0)
		{
			for (double& v : dir)
				v = -v;
			return;
		}

		std::vector<double> alpha(m, 0.0);
		std::vector<double> rho(m, 0.0);

		for (int i = 0; i < m; ++i)
			rho[i] = (std::abs(history[i].ys) > 1e-30) ? (1.0 / history[i].ys) : 0.0;

		for (int i = m - 1; i >= 0; --i)
		{
			alpha[i] = rho[i] * Dot(history[i].s, dir);
			AddScaled(dir, history[i].y, -alpha[i]);
		}

		double gamma = 1.0;
		{
			const HistoryPair& last = history.back();
			const double yy = Dot(last.y, last.y);
			if (yy > 1e-30)
				gamma = last.ys / yy;
		}

		for (double& v : dir)
			v *= gamma;

		for (int i = 0; i < m; ++i)
		{
			const double beta = rho[i] * Dot(history[i].y, dir);
			AddScaled(dir, history[i].s, alpha[i] - beta);
		}

		for (double& v : dir)
			v = -v;
	}

	bool ArmijoBacktracking(
		const std::vector<double>& x,
		double fx,
		const std::vector<double>& grad,
		const std::vector<double>& dir,
		std::vector<double>& xNew,
		double& fNew,
		double& usedStep)
	{
		const double c1 = 1e-4;
		const double dg0 = Dot(grad, dir);

		xNew = x;
		fNew = fx;
		usedStep = 0.0;

		if (!(dg0 < 0.0))
			return false;

		double step = std::min(std::max(lr, minStep), maxStep);

		for (int iter = 0; iter < 28; ++iter)
		{
			for (int i = 0; i < numPoints; ++i)
				xNew[i] = x[i] + step * dir[i];

			fNew = EvalError(xNew);

			if (fNew <= fx + c1 * step * dg0)
			{
				usedStep = step;
				return true;
			}

			step *= refineStepShrink;
			if (step < minStep)
				break;
		}

		xNew = x;
		fNew = fx;
		usedStep = 0.0;
		return false;
	}

	bool TryStep(
		const std::vector<double>& x,
		double fx,
		const std::vector<double>& grad,
		std::vector<double>& dir,
		std::vector<double>& xNew,
		double& fNew,
		double& usedStep)
	{
		if (ArmijoBacktracking(x, fx, grad, dir, xNew, fNew, usedStep))
			return true;

		history.clear();

		dir = grad;
		for (double& v : dir)
			v = -v;

		if (ArmijoBacktracking(x, fx, grad, dir, xNew, fNew, usedStep))
			return true;

		return false;
	}

	void PushHistory(const std::vector<double>& s, const std::vector<double>& y)
	{
		const double ys = Dot(y, s);
		const double ss = Dot(s, s);
		const double yy = Dot(y, y);

		const double quality = 1e-8 * sqrt(std::max(1e-30, ss * yy));

		if (ys > quality)
		{
			if ((int)history.size() >= historySize)
				history.erase(history.begin());
			history.push_back({ s, y, ys });
		}
		else
		{
			history.clear();
		}
	}

	void RefreshBestFromNow()
	{
		if (err < besterr)
		{
			besterr = err;
			bestVec = nowVec;
			bestCycleAt = cycles;
		}
	}

	void ResetTrackingState()
	{
		stallCount = 0;
		lineSearchFailCount = 0;
		noImproveCount = 0;
		convergedHint = false;
		history.clear();
	}

public:
	LbfgsOptimizer3() = default;

	void SetErrorFunc(ErrorFuncType errorfunc) final override
	{
		this->errfunc = errorfunc;
	}

	void SetupOptimizer(int numPoints, std::vector<float> initVec, float learningRate) final override
	{
		this->numPoints = numPoints;
		this->lr = std::max(1e-12, (double)learningRate);

		nowVec = std::move(initVec);
		nowVec.resize(numPoints, 0.0f);

		bestVec = nowVec;
		nowVecCopy.resize(numPoints, 0.0f);
		gradVec.assign(numPoints, 0.0f);

		err = 1e300;
		besterr = 1e300;
		cycles = 0;
		bestCycleAt = 0;

		ResetTrackingState();

		std::vector<double> x;
		CopyFloatToDouble(nowVec, x);
		err = EvalError(x);
		besterr = err;
		bestVec = nowVec;
		bestCycleAt = cycles;
	}

	void SetBasin(std::vector<float>& best) final override
	{
		nowVec = best;
		nowVec.resize(numPoints, 0.0f);
		bestVec = nowVec;
		nowVecCopy.resize(numPoints, 0.0f);
		gradVec.assign(numPoints, 0.0f);

		ResetTrackingState();

		lr = std::min(std::max(basinInitStep, minStep), maxStep);

		std::vector<double> x;
		CopyFloatToDouble(nowVec, x);
		err = EvalError(x);
		besterr = err;
		bestCycleAt = cycles;
	}

	void SetPersistentMode(bool enable)
	{
		persistentMode = enable;
	}

	bool HasConvergedHint() const
	{
		return convergedHint;
	}

	void SetHistorySize(int newSize)
	{
		historySize = std::max(1, newSize);
		if ((int)history.size() > historySize)
			history.erase(history.begin(), history.end() - historySize);
	}

	void SetFiniteDiffRelStep(double v)
	{
		fdRelStep = std::max(1e-12, v);
	}

	void SetStepRange(double minV, double maxV)
	{
		minStep = std::max(1e-16, minV);
		maxStep = std::max(minStep, maxV);
		tinyStepTol = std::max(minStep, tinyStepTol);
	}

	void SetRefineStepControl(double shrink, double grow)
	{
		refineStepShrink = std::min(0.95, std::max(0.1, shrink));
		refineStepGrow = std::max(1.0, grow);
	}

	void SetTolerances(double gradTolV, double relImproveTolV)
	{
		gradTol = std::max(1e-16, gradTolV);
		relImproveTol = std::max(0.0, relImproveTolV);
	}

	void RunOptimizer(int numCycles) final override
	{
		if (numPoints <= 0 || numCycles <= 0)
			return;

		std::vector<double> x;
		std::vector<double> g;
		std::vector<double> xNew;
		std::vector<double> gNew;
		std::vector<double> dir;

		CopyFloatToDouble(nowVec, x);
		xNew.resize(numPoints);
		g.resize(numPoints);
		gNew.resize(numPoints);
		dir.resize(numPoints);

		err = EvalError(x);
		RefreshBestFromNow();
		EvalGradient(x, err, g);
		CopyDoubleToFloat(g, gradVec);

		for (int iter = 0; iter < numCycles; ++iter, ++cycles)
		{
			const double gnorm = Norm2(g);
			if (gnorm < gradTol)
				convergedHint = true;

			ComputeLbfgsDirection(g, dir);

			double fNew = err;
			double usedStep = 0.0;
			bool ok = TryStep(x, err, g, dir, xNew, fNew, usedStep);

			if (!ok)
			{
				++lineSearchFailCount;
				++stallCount;
				++noImproveCount;

				history.clear();
				lr = std::max(minStep, lr * refineStepShrink);

				if (lr <= tinyStepTol)
					convergedHint = true;

				if (!persistentMode && lineSearchFailCount >= lineSearchFailLimit)
					break;

				EvalGradient(x, err, g);
				CopyDoubleToFloat(g, gradVec);
				continue;
			}

			lineSearchFailCount = 0;

			EvalGradient(xNew, fNew, gNew);

			std::vector<double> s = Sub(xNew, x);
			std::vector<double> y = Sub(gNew, g);
			PushHistory(s, y);

			x = xNew;
			g = gNew;
			err = fNew;

			CopyDoubleToFloat(x, nowVec);
			CopyDoubleToFloat(g, gradVec);

			if (err < besterr)
			{
				const double prevBest = besterr;
				besterr = err;
				bestVec = nowVec;
				bestCycleAt = cycles;

				const double denom = std::max(1.0, std::abs(prevBest));
				const double relImprove = (prevBest - err) / denom;

				if (relImprove > relImproveTol)
				{
					stallCount = 0;
					noImproveCount = 0;
				}
				else
				{
					++stallCount;
					++noImproveCount;
				}
			}
			else
			{
				++stallCount;
				++noImproveCount;
			}

			lr = std::min(maxStep, std::max(minStep, usedStep * refineStepGrow));

			if (usedStep <= tinyStepTol)
				convergedHint = true;

			if (!persistentMode)
			{
				if (stallCount >= stallLimit || noImproveCount >= noImproveLimit)
					break;
			}
		}
	}

	int GetBestVec(std::vector<float>& best) final override
	{
		best = bestVec;
		return bestCycleAt;
	}

	int GetNowVec(std::vector<float>& nowv) final override
	{
		nowv = nowVec;
		return cycles;
	}

	float GetNowError() final override
	{
		return (float)err;
	}
};

class AdamOptimizer :public OptimizerBase
{
private:
	using ErrorFuncType = std::function<float(std::vector<float>&)>;
	ErrorFuncType errfunc = [](std::vector<float>&) { return 0.0f; };

	int numPoints = 0;

	std::vector<float> bestVec;
	std::vector<float> nowVec;
	std::vector<float> nowVecCopy;
	std::vector<float> gradVec;

	std::vector<float> mVec;
	std::vector<float> vVec;

	float lr = 0.001f;
	float beta1 = 0.9f;
	float beta2 = 0.999f;
	float eps = 1e-8f;

	float err = 1e20f;
	float besterr = 1e20f;

	int cycles = 0;
	int bestCycleAt = 0;

private:
	float EvalError(std::vector<float>& x)
	{
		return errfunc(x);
	}

	void EvalGradient(const std::vector<float>& x, float fx, std::vector<float>& g)
	{
		g.resize(numPoints);
		nowVecCopy = x;

		for (int j = 0; j < numPoints; ++j)
		{
			const float base = x[j];
			const float h = 1e-4f * std::max(1.0f, std::abs(base));

			nowVecCopy[j] = base + h;
			const float f1 = EvalError(nowVecCopy);

			nowVecCopy[j] = base - h;
			const float f2 = EvalError(nowVecCopy);

			g[j] = (f1 - f2) / (2.0f * h);
			nowVecCopy[j] = base;
		}
	}

public:
	AdamOptimizer() = default;

	void SetErrorFunc(ErrorFuncType errorfunc)final override
	{
		this->errfunc = errorfunc;
	}

	void SetupOptimizer(int numPoints, std::vector<float> initVec, float learningRate)final override
	{
		this->numPoints = numPoints;
		this->lr = learningRate;

		nowVec = initVec;
		nowVec.resize(numPoints, 0.0f);

		bestVec = nowVec;
		nowVecCopy.resize(numPoints, 0.0f);
		gradVec.resize(numPoints, 0.0f);

		mVec.assign(numPoints, 0.0f);
		vVec.assign(numPoints, 0.0f);

		err = 1e20f;
		besterr = 1e20f;
		cycles = 0;
		bestCycleAt = 0;
	}

	void SetBasin(std::vector<float>& best)final override
	{
		nowVec = best;
		bestVec = best;
	}
	void RunOptimizer(int numCycles)final override
	{
		if (numPoints <= 0) return;

		for (int i = 0; i < numCycles; ++i)
		{
			err = EvalError(nowVec);

			if (err < besterr)
			{
				besterr = err;
				bestVec = nowVec;
				bestCycleAt = cycles;
			}

			EvalGradient(nowVec, err, gradVec);

			const float t = float(cycles + 1);
			const float beta1Pow = pow(beta1, t);
			const float beta2Pow = pow(beta2, t);

			for (int j = 0; j < numPoints; ++j)
			{
				const float g = gradVec[j];

				mVec[j] = beta1 * mVec[j] + (1.0f - beta1) * g;
				vVec[j] = beta2 * vVec[j] + (1.0f - beta2) * g * g;

				const float mHat = mVec[j] / (1.0f - beta1Pow);
				const float vHat = vVec[j] / (1.0f - beta2Pow);

				nowVec[j] -= lr * mHat / (sqrt(vHat) + eps);
			}

			++cycles;
		}

		err = EvalError(nowVec);
		if (err < besterr)
		{
			besterr = err;
			bestVec = nowVec;
			bestCycleAt = cycles;
		}
	}

	int GetBestVec(std::vector<float>& best)final override
	{
		best = bestVec;
		return bestCycleAt;
	}

	int GetNowVec(std::vector<float>& nowv)final override
	{
		nowv = nowVec;
		return cycles;
	}

	float GetNowError()final override
	{
		return err;
	}

	void SetAdamParams(float beta1_, float beta2_, float eps_)
	{
		beta1 = beta1_;
		beta2 = beta2_;
		eps = eps_;
	}
};