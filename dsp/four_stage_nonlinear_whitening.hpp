#pragma once

#include <vector>
#include <algorithm>

namespace FourStageNonlinearWhitening
{

    static constexpr int kDim = 9;
    static constexpr int kSplineKnots = 17;

    static const double kWhitenMu[9] = {
        -0.0084689832658573407,
        -0.013218043666521975,
        0.04197484597297467,
        -0.0095900958886217611,
        -0.0080500556219777648,
        0.0084304750623102609,
        0.011879087681416253,
        0.014883935913255136,
        -0.0020431403995212691,
    };

    static const double kWhitenR[9][9] = {
        { 21.152290695743879, 1.6780624842678067, 5.4159916569023396, -6.8429111124620903, 2.6051954038693377, 0.010541327959420012, -0.024885948002281937, -0.0068374537102261993, -0.028579555929359797 },
        { 1.6780624842678065, 18.391540330625055, -1.2536947647870647, 7.4162473494076755, -3.8098302607581696, -0.04809785281753233, -0.0087088767418576855, -0.010381046737067946, 0.014234125786835805 },
        { 5.4159916569023387, -1.2536947647870644, 18.082662271547111, 1.9526310729343261, 9.833777629732543, -0.0065803528840592037, -0.035006358506685077, 0.01294935406354052, -0.010903836058414157 },
        { -6.8429111124620903, 7.4162473494076755, 1.9526310729343261, 21.914862838936596, 4.2236247845902612, -0.009946018571171486, 0.014179788868569942, -0.038827625036344328, 0.0026466467718049011 },
        { 2.6051954038693377, -3.8098302607581687, 9.8337776297325448, 4.2236247845902612, 21.281949657267891, 0.013741725660668606, -0.013316986464677088, 0.010824689991378396, -0.0336265593187016 },
        { 0.010541327959420053, -0.048097852817532274, -0.0065803528840591109, -0.0099460185711714669, 0.013741725660668669, 21.119591232904732, 0.66457813048100156, 10.428311523862536, -2.1488505134414191 },
        { -0.024885948002281975, -0.008708876741857741, -0.035006358506685174, 0.014179788868569968, -0.013316986464677145, 0.66457813048100178, 21.255586362255638, -0.55116939349142735, 10.179650688792975 },
        { -0.0068374537102261343, -0.010381046737067964, 0.012949354063540532, -0.038827625036344321, 0.010824689991378386, 10.428311523862536, -0.55116939349142879, 21.007812703817837, 0.68296537619919606 },
        { -0.02857955592935979, 0.014234125786835783, -0.010903836058414189, 0.0026466467718048929, -0.033626559318701586, -2.1488505134414186, 10.179650688792973, 0.68296537619919684, 21.375031578939087 },
    };

    static const double kWhitenRInv[9][9] = {
        { 0.065193905876703317, -0.021732094533711069, -0.019472492767888257, 0.031193265833664763, -0.009063869595264392, -0.00013463565089531979, -1.2339719534353188e-05, 0.00014932743930741961, 6.1155359890858481e-05 },
        { -0.021732094533711065, 0.076106917235655713, 0.00391609269805569, -0.03709924481917657, 0.02183785830415785, 0.00022811514099811169, 5.751246938706247e-05, -0.00016218990818074681, -3.8067283947451604e-05 },
        { -0.019472492767888257, 0.0039160926980556909, 0.079774862608503763, -0.0083222487565728122, -0.032125364443338356, 9.279960846149594e-05, 0.0001326370402085691, -9.2118338636736844e-05, -8.8351559045463752e-05 },
        { 0.031193265833664763, -0.03709924481917657, -0.0083222487565728104, 0.072724331326443972, -0.021047303853080682, -0.00015868536766483032, -5.449622658128563e-05, 0.00021879950083881559, 2.3060806716696844e-05 },
        { -0.009063869595264392, 0.021837858304157857, -0.032125364443338356, -0.021047303853080682, 0.071028468148350496, 3.372371834835411e-05, -4.6249087964493969e-05, -6.9025726936030316e-05, 9.8918349778564406e-05 },
        { -0.00013463565089531969, 0.00022811514099811179, 9.279960846149594e-05, -0.00015868536766483037, 3.3723718348354205e-05, 0.065033369216346337, -0.0084463644273483161, -0.032882035952257485, 0.011610780103483364 },
        { -1.2339719534353176e-05, 5.7512469387062436e-05, 0.00013263704020856915, -5.4496226581285562e-05, -4.6249087964494003e-05, -0.0084463644273483161, 0.062180903556799748, 0.0068214722258604487, -0.030680180059505055 },
        { 0.00014932743930741964, -0.00016218990818074675, -9.2118338636736709e-05, 0.00021879950083881542, -6.9025726936030303e-05, -0.032882035952257478, 0.0068214722258604504, 0.064383424550512006, -0.0086113445799333426 },
        { 6.1155359890858508e-05, -3.8067283947451584e-05, -8.8351559045463779e-05, 2.3060806716696872e-05, 9.8918349778564515e-05, 0.011610780103483364, -0.030680180059505055, -0.0086113445799333409, 0.062837296104977597 },
    };

    static const double kSplineX_0[17] = {
        -6,
        -5.25,
        -4.5,
        -3.75,
        -3,
        -2.25,
        -1.5,
        -0.75,
        0,
        0.75,
        1.5,
        2.25,
        3,
        3.75,
        4.5,
        5.25,
        6,
    };

    static const double kSplineY_0[17] = {
        -4.9909402507723106,
        -1.9399124580195326,
        1.1181413512001406,
        3.9608223632309221,
        6.1094545026691671,
        6.839886045133686,
        7.2751360105924201,
        7.6929763200701773,
        8.1103073894857367,
        8.5276979428848385,
        8.9502371427983363,
        9.3800045202483062,
        9.8171948227950807,
        10.203185273853673,
        10.314809118420083,
        10.316101954839752,
        10.316458962702455,
    };

    static constexpr double kSplineLeftSlope_0 = 1;
    static constexpr double kSplineRightSlope_0 = 1;

    static const double kSplineX_1[17] = {
        -6,
        -5.25,
        -4.5,
        -3.75,
        -3,
        -2.25,
        -1.5,
        -0.75,
        0,
        0.75,
        1.5,
        2.25,
        3,
        3.75,
        4.5,
        5.25,
        6,
    };

    static const double kSplineY_1[17] = {
        -5.9418793275682438,
        -5.126660954084306,
        -4.3113732763630797,
        -3.5001040969086268,
        -2.7520262388121535,
        -2.2019527223163209,
        -1.7966136614733852,
        -1.4268716959181162,
        -1.0617328059345414,
        -0.70279226617021351,
        -0.3531396028095708,
        -0.023160908992720941,
        0.30245977824881987,
        0.65545966024805224,
        1.0188699882923808,
        2.6488622379244502,
        6.0004714398871188,
    };

    static constexpr double kSplineLeftSlope_1 = 1;
    static constexpr double kSplineRightSlope_1 = 1;

    static const double kSplineX_2[17] = {
        -6,
        -5.25,
        -4.5,
        -3.75,
        -3,
        -2.25,
        -1.5,
        -0.75,
        0,
        0.75,
        1.5,
        2.25,
        3,
        3.75,
        4.5,
        5.25,
        6,
    };

    static const double kSplineY_2[17] = {
        -5.1109642975419041,
        -3.0526699851428889,
        -0.99160132888543639,
        0.93430771021849424,
        2.4670085506796573,
        3.0082062339711504,
        3.2823126495686177,
        3.5459654508179108,
        3.8150088672532485,
        4.0918710427396254,
        4.3807796495213767,
        4.682705738930534,
        4.9809604901391191,
        5.3248766696445129,
        5.3560939071308136,
        5.3600019363747906,
        5.3617385566942461,
    };

    static constexpr double kSplineLeftSlope_2 = 1;
    static constexpr double kSplineRightSlope_2 = 1;

    static const double kSplineX_3[17] = {
        -6,
        -5.25,
        -4.5,
        -3.75,
        -3,
        -2.25,
        -1.5,
        -0.75,
        0,
        0.75,
        1.5,
        2.25,
        3,
        3.75,
        4.5,
        5.25,
        6,
    };

    static const double kSplineY_3[17] = {
        -5.3744253413317642,
        -3.7471178426750695,
        -2.1203093779803335,
        -0.51341105543079024,
        0.91679481899871185,
        1.7897329439331591,
        2.3678589735818578,
        2.8696016857542981,
        3.3523741803967857,
        3.8249713076033034,
        4.2892853035687954,
        4.7312042439601578,
        5.1813775896842467,
        5.6721116558227491,
        6.1070197687068299,
        6.4310594739730069,
        6.4325398965309608,
    };

    static constexpr double kSplineLeftSlope_3 = 1;
    static constexpr double kSplineRightSlope_3 = 1;

    static const double kSplineX_4[17] = {
        -6,
        -5.25,
        -4.5,
        -3.75,
        -3,
        -2.25,
        -1.5,
        -0.75,
        0,
        0.75,
        1.5,
        2.25,
        3,
        3.75,
        4.5,
        5.25,
        6,
    };

    static const double kSplineY_4[17] = {
        -8.3854457483829918,
        -8.1573203398655796,
        -7.9292449414126533,
        -7.7008548615335739,
        -7.4613640810267254,
        -7.1585048238088564,
        -6.7961477408069646,
        -6.4300772415887435,
        -6.0729462812073294,
        -5.7230710964275682,
        -5.3789159934073183,
        -5.0345623502488177,
        -4.6928129884608385,
        -4.3668448215100595,
        2.3079560200233296,
        5.2749008819003755,
        6.0042595876465761,
    };

    static constexpr double kSplineLeftSlope_4 = 1;
    static constexpr double kSplineRightSlope_4 = 1;

    static const double kSplineX_5[17] = {
        -6,
        -5.25,
        -4.5,
        -3.75,
        -3,
        -2.25,
        -1.5,
        -0.75,
        0,
        0.75,
        1.5,
        2.25,
        3,
        3.75,
        4.5,
        5.25,
        6,
    };

    static const double kSplineY_5[17] = {
        -5.8673030956665633,
        -5.1151911263052057,
        -4.3631210255985469,
        -3.6117854635059334,
        -2.8561297837264377,
        -2.2776928255232525,
        -1.8826639021536193,
        -1.5010145938050261,
        -1.1030269998279421,
        -0.68936723789293719,
        -0.26278860461290954,
        0.17279394332911746,
        0.63124552706753079,
        1.0980507410090956,
        1.2466458139611367,
        3.9516572247775175,
        6.0093385732764881,
    };

    static constexpr double kSplineLeftSlope_5 = 1;
    static constexpr double kSplineRightSlope_5 = 1;

    static const double kSplineX_6[17] = {
        -6,
        -5.25,
        -4.5,
        -3.75,
        -3,
        -2.25,
        -1.5,
        -0.75,
        0,
        0.75,
        1.5,
        2.25,
        3,
        3.75,
        4.5,
        5.25,
        6,
    };

    static const double kSplineY_6[17] = {
        -6.5223131442312896,
        -6.0530612540140973,
        -5.5839468872146574,
        -5.1131985118634855,
        -4.6344222708834408,
        -4.1233404044805484,
        -3.6195183086218403,
        -3.1163863628646729,
        -2.6254821622524256,
        -2.1530794163578229,
        -1.6969313886934287,
        -1.2385120347727891,
        -0.82120086181768226,
        -0.45617916083802079,
        0.05982752840906258,
        5.2334405884630071,
        5.9822617446558484,
    };

    static constexpr double kSplineLeftSlope_6 = 1;
    static constexpr double kSplineRightSlope_6 = 1;

    static const double kSplineX_7[17] = {
        -6,
        -5.25,
        -4.5,
        -3.75,
        -3,
        -2.25,
        -1.5,
        -0.75,
        0,
        0.75,
        1.5,
        2.25,
        3,
        3.75,
        4.5,
        5.25,
        6,
    };

    static const double kSplineY_7[17] = {
        -6.1906614113895326,
        -5.6148253596783189,
        -5.0389725794816744,
        -4.4627385235302821,
        -3.8998300740317005,
        -3.3688424258733654,
        -2.8878956869692094,
        -2.4402841549841514,
        -2.0075583219276885,
        -1.5947130090586761,
        -1.2146222535697531,
        -0.86835541482938972,
        -0.53695467657477636,
        -0.1135188619818619,
        0.19943071935149526,
        5.2409169533727198,
        5.9893540421257381,
    };

    static constexpr double kSplineLeftSlope_7 = 1;
    static constexpr double kSplineRightSlope_7 = 1;

    static const double kSplineX_8[17] = {
        -6,
        -5.25,
        -4.5,
        -3.75,
        -3,
        -2.25,
        -1.5,
        -0.75,
        0,
        0.75,
        1.5,
        2.25,
        3,
        3.75,
        4.5,
        5.25,
        6,
    };

    static const double kSplineY_8[17] = {
        -4.6573936261496627,
        -0.73064364630331191,
        3.2059178837223872,
        7.1309834144936648,
        10.740864001686655,
        12.644738739454461,
        13.141018433849236,
        13.556830723547058,
        13.989108317327453,
        14.442358539979917,
        14.913640550072234,
        15.392215327470112,
        15.866731091517412,
        16.321105145904461,
        16.412128052177192,
        16.412424984109002,
        16.412737047691817,
    };

    static constexpr double kSplineLeftSlope_8 = 1;
    static constexpr double kSplineRightSlope_8 = 1;


    inline double EvalSpline1D(
        double x,
        const double* xk,
        const double* yk,
        int knots,
        double leftSlope,
        double rightSlope)
    {
        if (x <= xk[0])
            return yk[0] + leftSlope * (x - xk[0]);

        if (x >= xk[knots - 1])
            return yk[knots - 1] + rightSlope * (x - xk[knots - 1]);

        int idx = 0;
        while (idx + 1 < knots && x > xk[idx + 1])
            ++idx;

        const double x0 = xk[idx];
        const double x1 = xk[idx + 1];
        const double y0 = yk[idx];
        const double y1 = yk[idx + 1];

        const double t = (x - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    inline double EvalSpline1DInverse(
        double y,
        const double* xk,
        const double* yk,
        int knots,
        double leftSlope,
        double rightSlope)
    {
        if (y <= yk[0])
            return xk[0] + (y - yk[0]) / leftSlope;

        if (y >= yk[knots - 1])
            return xk[knots - 1] + (y - yk[knots - 1]) / rightSlope;

        int idx = 0;
        while (idx + 1 < knots && y > yk[idx + 1])
            ++idx;

        const double y0 = yk[idx];
        const double y1 = yk[idx + 1];
        const double x0 = xk[idx];
        const double x1 = xk[idx + 1];

        const double t = (y - y0) / (y1 - y0);
        return x0 + t * (x1 - x0);
    }

    inline void MulMat9x9Vec9(const double M[9][9], const double* x, double* y)
    {
        for (int i = 0; i < 9; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < 9; ++j)
                sum += M[i][j] * x[j];
            y[i] = sum;
        }
    }

    inline void ForwardVToA(const std::vector<float>& vin, std::vector<float>& aout)
    {
        double v[9] = {};
        double z[9] = {};
        double a0[9] = {};

        for (int i = 0; i < 9 && i < (int)vin.size(); ++i)
            v[i] = (double)vin[i];

        z[0] = EvalSpline1D(v[0], kSplineX_0, kSplineY_0, kSplineKnots, kSplineLeftSlope_0, kSplineRightSlope_0);
        z[1] = EvalSpline1D(v[1], kSplineX_1, kSplineY_1, kSplineKnots, kSplineLeftSlope_1, kSplineRightSlope_1);
        z[2] = EvalSpline1D(v[2], kSplineX_2, kSplineY_2, kSplineKnots, kSplineLeftSlope_2, kSplineRightSlope_2);
        z[3] = EvalSpline1D(v[3], kSplineX_3, kSplineY_3, kSplineKnots, kSplineLeftSlope_3, kSplineRightSlope_3);
        z[4] = EvalSpline1D(v[4], kSplineX_4, kSplineY_4, kSplineKnots, kSplineLeftSlope_4, kSplineRightSlope_4);
        z[5] = EvalSpline1D(v[5], kSplineX_5, kSplineY_5, kSplineKnots, kSplineLeftSlope_5, kSplineRightSlope_5);
        z[6] = EvalSpline1D(v[6], kSplineX_6, kSplineY_6, kSplineKnots, kSplineLeftSlope_6, kSplineRightSlope_6);
        z[7] = EvalSpline1D(v[7], kSplineX_7, kSplineY_7, kSplineKnots, kSplineLeftSlope_7, kSplineRightSlope_7);
        z[8] = EvalSpline1D(v[8], kSplineX_8, kSplineY_8, kSplineKnots, kSplineLeftSlope_8, kSplineRightSlope_8);

        MulMat9x9Vec9(kWhitenRInv, z, a0);

        aout.resize(9);
        for (int i = 0; i < 9; ++i)
            aout[i] = (float)(a0[i] + kWhitenMu[i]);
    }

    inline void InverseAToV(const std::vector<float>& ain, std::vector<float>& vout)
    {
        double a0[9] = {};
        double z[9] = {};
        double v[9] = {};

        for (int i = 0; i < 9 && i < (int)ain.size(); ++i)
            a0[i] = (double)ain[i] - kWhitenMu[i];

        MulMat9x9Vec9(kWhitenR, a0, z);

        v[0] = EvalSpline1DInverse(z[0], kSplineX_0, kSplineY_0, kSplineKnots, kSplineLeftSlope_0, kSplineRightSlope_0);
        v[1] = EvalSpline1DInverse(z[1], kSplineX_1, kSplineY_1, kSplineKnots, kSplineLeftSlope_1, kSplineRightSlope_1);
        v[2] = EvalSpline1DInverse(z[2], kSplineX_2, kSplineY_2, kSplineKnots, kSplineLeftSlope_2, kSplineRightSlope_2);
        v[3] = EvalSpline1DInverse(z[3], kSplineX_3, kSplineY_3, kSplineKnots, kSplineLeftSlope_3, kSplineRightSlope_3);
        v[4] = EvalSpline1DInverse(z[4], kSplineX_4, kSplineY_4, kSplineKnots, kSplineLeftSlope_4, kSplineRightSlope_4);
        v[5] = EvalSpline1DInverse(z[5], kSplineX_5, kSplineY_5, kSplineKnots, kSplineLeftSlope_5, kSplineRightSlope_5);
        v[6] = EvalSpline1DInverse(z[6], kSplineX_6, kSplineY_6, kSplineKnots, kSplineLeftSlope_6, kSplineRightSlope_6);
        v[7] = EvalSpline1DInverse(z[7], kSplineX_7, kSplineY_7, kSplineKnots, kSplineLeftSlope_7, kSplineRightSlope_7);
        v[8] = EvalSpline1DInverse(z[8], kSplineX_8, kSplineY_8, kSplineKnots, kSplineLeftSlope_8, kSplineRightSlope_8);

        vout.resize(9);
        for (int i = 0; i < 9; ++i)
            vout[i] = (float)v[i];
    }

} // namespace FourStageNonlinearWhitening
