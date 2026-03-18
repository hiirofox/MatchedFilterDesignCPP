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
    -5.9219295373794747,
    -5.1109735558927705,
    -4.3000175743368985,
    -3.4890615970231074,
    -2.6785567654932336,
    -1.8685066145286244,
    -1.0625192335735418,
    -0.28608388754654079,
    0.47315487748416629,
    1.2572452930419393,
    2.0708287939385563,
    2.8607552705547619,
    3.609015822056314,
    4.3259623113425647,
    5.0810283467697896,
    5.7688876971705696,
    6.4839585647825828,
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
    -6.0379532051271072,
    -5.3160413084982787,
    -4.594129411918356,
    -3.8722179085948492,
    -3.1501823825778663,
    -2.4281137936874693,
    -1.713663386238907,
    -0.99788628500419385,
    -0.26318308279426006,
    0.4776236143485395,
    1.2500250654155183,
    1.9922572374757772,
    2.7438289009330186,
    3.4601321812164469,
    4.1816792872161299,
    4.9675125855104794,
    5.752699935685631,
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
    -5.9875591549702651,
    -5.2290058523016141,
    -4.4704525506997292,
    -3.7118889140899083,
    -2.9534638846105326,
    -2.1971428000446513,
    -1.4439407402076379,
    -0.68415058818553653,
    0.092846722772561563,
    0.83214757545025186,
    1.6168549748203676,
    2.3660506776525274,
    3.0766298562090508,
    3.8302694795565699,
    4.5605718688578438,
    5.3029703710606508,
    6.0492089227069616,
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
    -5.9636965449392596,
    -5.1862401919223844,
    -4.4087838388977785,
    -3.6313274910883395,
    -2.8532728625702539,
    -2.0712883431923479,
    -1.2883344364753233,
    -0.49840299904741681,
    0.27941271180978955,
    1.0407055302633195,
    1.784621444644066,
    2.5600429641226752,
    3.2860388226307462,
    4.004543527319723,
    4.6909945128529609,
    5.4456817458368585,
    6.198188633192026,
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
    -5.9466494411852899,
    -5.1558894745813841,
    -4.3651295253220672,
    -3.5743697014140281,
    -2.7831522421847237,
    -1.9921187542062628,
    -1.2014825434784875,
    -0.41937990927785052,
    0.39823125570531914,
    1.1802890761871865,
    1.890245756414684,
    2.5979958659352533,
    3.2985332320452052,
    3.9080339655374292,
    4.61063300474774,
    5.3448906946413874,
    6.0942491796200873,
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
    -6.0748996659681849,
    -5.3780268480831621,
    -4.6811540465218107,
    -3.984005589297583,
    -3.2866372355465221,
    -2.5884483880467206,
    -1.8805898754106813,
    -1.1600331286815111,
    -0.40566782695709858,
    0.40338478349005324,
    1.1834671208730976,
    2.0027882506232411,
    2.759482957838272,
    3.5581680523210801,
    4.3480268818489849,
    5.1026577187174533,
    5.8610885207363586,
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
    -5.977862157402007,
    -5.2116991858608941,
    -4.4455361626939069,
    -3.6793507309610334,
    -2.9128894532315166,
    -2.1466010786426875,
    -1.3795369679527667,
    -0.60144526652066332,
    0.18256478766728979,
    0.99041037387037001,
    1.8114909464841862,
    2.6061604636860993,
    3.39423141054134,
    4.1459223925455158,
    4.9422301473023618,
    5.6488626342051722,
    6.3754609209552049,
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
    -5.8919413560341658,
    -5.0553302235427049,
    -4.2187190691405467,
    -3.382109119708669,
    -2.5439096282732292,
    -1.7031097710526284,
    -0.86014545368511808,
    -0.007911233421310726,
    0.84189109235920245,
    1.6700677773602504,
    2.4756452796569315,
    3.2493593269154486,
    4.0386492369190021,
    4.7868559185263164,
    5.5808337128515246,
    6.2504067397717682,
    6.9537692998152005,
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
    -5.9993563330028783,
    -5.249527640483854,
    -4.4996989515170078,
    -3.7498677816558774,
    -3.00007987167404,
    -2.2535803028180443,
    -1.5097336429634547,
    -0.74896832128562529,
    0.035669618989171425,
    0.83699239466091058,
    1.656250028276407,
    2.5129338675113209,
    3.2857783343026226,
    4.0358930007827842,
    4.8111300789744993,
    5.5551645173835853,
    6.2850213300495659,
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
