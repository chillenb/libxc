
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_r2scan0_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.741291867882683e+01, -1.741297049485472e+01, -1.741323897784408e+01, -1.741246885561173e+01, -1.741286693937372e+01, -1.741286693937372e+01, -2.596476990726174e+00, -2.596568743463249e+00, -2.599183106664471e+00, -2.599838074409784e+00, -2.598885236976861e+00, -2.598885236976861e+00, -4.863283205387565e-01, -4.856579205083033e-01, -4.702049742570784e-01, -4.783616820039861e-01, -4.773663028836225e-01, -4.773663028836225e-01, -1.502125287426358e-01, -1.523179936312690e-01, -5.358335477985471e-01, -1.013069502282089e-01, -1.360248228822511e-01, -1.360248228822511e-01, -2.625997676137338e-03, -2.812257696562889e-03, -2.251650289975448e-02, -1.236296297256011e-03, -1.866200170499929e-03, -1.866200170499929e-03, -4.277416547147462e+00, -4.278253545539741e+00, -4.277473538949826e+00, -4.278211710133409e+00, -4.277831316924026e+00, -4.277831316924026e+00, -1.582195142537434e+00, -1.599227379815055e+00, -1.576859847403584e+00, -1.592070208628038e+00, -1.594340222095283e+00, -1.594340222095283e+00, -4.904899207528605e-01, -5.338113233696278e-01, -4.442523106764474e-01, -4.625038562726101e-01, -4.999361732903885e-01, -4.999361732903886e-01, -6.644524044207378e-02, -1.448208504889465e-01, -6.113959641020764e-02, -1.490972996089509e+00, -8.356900675576649e-02, -8.356900675576648e-02, -8.934498468369641e-04, -1.218982241243930e-03, -1.023951773788455e-03, -3.976453859319449e-02, -1.260653434672610e-03, -1.260653434672608e-03, -5.047322717069859e-01, -5.016117947319656e-01, -5.027071904822865e-01, -5.036117892662703e-01, -5.031582736944646e-01, -5.031582736944646e-01, -4.908215471721739e-01, -4.207097751134466e-01, -4.409158065336847e-01, -4.608532250061297e-01, -4.506056223008132e-01, -4.506056223008132e-01, -5.544003554324395e-01, -1.870991080320728e-01, -2.232746404207722e-01, -2.913282300892505e-01, -2.562971728862650e-01, -2.562971728862650e-01, -3.823636321662516e-01, -2.129237607824329e-02, -3.051105272161768e-02, -2.856482725856472e-01, -5.305438929987150e-02, -5.305438929987151e-02, -4.003694322881148e-03, -2.321899887160498e-04, -6.054867663211796e-04, -5.030702268212886e-02, -1.152284993965840e-03, -1.152284993965831e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_r2scan0_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.265172913825850e+01, -2.265180369007318e+01, -2.265182287474941e+01, -2.265187176012715e+01, -2.265212343613794e+01, -2.265230224913846e+01, -2.265106576974110e+01, -2.265095398790344e+01, -2.265178519028937e+01, -2.265149355569505e+01, -2.265178519028937e+01, -2.265149355569505e+01, -3.675016452100364e+00, -3.675100493846652e+00, -3.675097396377617e+00, -3.675165227884793e+00, -3.676760921454883e+00, -3.677214624385194e+00, -3.675379453044178e+00, -3.675636392670854e+00, -3.673908424196218e+00, -3.677531075966579e+00, -3.673908424196218e+00, -3.677531075966579e+00, -6.932776415574196e-01, -6.963925117562429e-01, -6.918105875545476e-01, -6.954612435530203e-01, -6.681927698648252e-01, -6.647762353192348e-01, -6.768635908243962e-01, -6.789084898452794e-01, -6.939571176899202e-01, -6.542992644816448e-01, -6.939571176899202e-01, -6.542992644816448e-01, -1.672225564554903e-01, -1.913750878829628e-01, -1.728907338682466e-01, -1.955200042159902e-01, -7.165717265975777e-01, -7.437956113250235e-01, -1.430054749695338e-01, -1.290381466346814e-01, -1.536383691497312e-01, -1.398826558402489e-01, -1.536383691497311e-01, -1.398826558402489e-01, -5.899038734223171e-03, 6.257556281081204e-03, -5.248535199968193e-03, -3.410095638309599e-03, -3.628951280354680e-02, -3.707146416100361e-02, -2.323005009071058e-03, -2.287723248915125e-03, -3.463225369834231e-03, -2.517518995707347e-03, -3.463225369834241e-03, -2.517518995707346e-03, -5.592504405878568e+00, -5.591182159720793e+00, -5.594304426864327e+00, -5.592919949452507e+00, -5.592614038272349e+00, -5.591247966107946e+00, -5.594157483943304e+00, -5.592827160454386e+00, -5.593419425815077e+00, -5.592054185699788e+00, -5.593419425815077e+00, -5.592054185699788e+00, -1.960616132189969e+00, -1.962291813044082e+00, -1.986578303353904e+00, -1.987436998968216e+00, -1.942658260684944e+00, -1.949030078816295e+00, -1.966532973204262e+00, -1.972946907213407e+00, -1.988868654396815e+00, -1.977893685249192e+00, -1.988868654396815e+00, -1.977893685249192e+00, -6.553850887844558e-01, -6.540748758272033e-01, -7.208435287352134e-01, -7.210009927407806e-01, -6.014034840677556e-01, -6.203285154199436e-01, -6.379871400722625e-01, -6.523426902214293e-01, -6.795230723035546e-01, -6.521463945602499e-01, -6.795230723035547e-01, -6.521463945602499e-01, -9.607784562087521e-02, -9.785756557253686e-02, -1.416268463937435e-01, -1.425552459236903e-01, -9.067619689648650e-02, -9.082742191134612e-02, -2.154547866809227e+00, -2.153803774201597e+00, -9.262066018595549e-02, -8.244735632964548e-02, -9.262066018595815e-02, -8.244735632964513e-02, -1.660589601996555e-03, -1.693420362423074e-03, -2.266435261887900e-03, -2.280768483457351e-03, -1.855159573438681e-03, -1.943104935559807e-03, -6.265797689740668e-02, -6.080823473359524e-02, -1.972601847825818e-03, -2.269136510556452e-03, -1.972601847825814e-03, -2.269136510556448e-03, -6.656393413955797e-01, -6.678578225743916e-01, -6.600178500126748e-01, -6.623026984358659e-01, -6.620232863294000e-01, -6.643029635336394e-01, -6.636794155859609e-01, -6.659066984923286e-01, -6.628552901354952e-01, -6.651079451291084e-01, -6.628552901354952e-01, -6.651079451291084e-01, -6.491840674841606e-01, -6.509053225240166e-01, -5.424537771987122e-01, -5.447554709634708e-01, -5.738516832274508e-01, -5.763269252724421e-01, -6.045026291767730e-01, -6.063558601877499e-01, -5.891889947789422e-01, -5.911629523474768e-01, -5.891889947789422e-01, -5.911629523474768e-01, -7.558435527863421e-01, -7.565269296667549e-01, -2.122539377662836e-01, -2.134726743227225e-01, -2.703080517174974e-01, -2.744312818246146e-01, -3.901319614606003e-01, -3.918798188530177e-01, -3.289913649090072e-01, -3.293370268648885e-01, -3.289913649090072e-01, -3.293370268648886e-01, -4.980912176483767e-01, -5.023273081376666e-01, -3.186330778515797e-02, -3.373754696409229e-02, -4.840119882660025e-02, -4.917118041327380e-02, -3.810837112291588e-01, -3.860079864460449e-01, -7.704095000108933e-02, -7.673834140537651e-02, -7.704095000108933e-02, -7.673834140537651e-02, -7.113971606116140e-03, -7.302106697043317e-03, -4.354445964392907e-04, -4.539807524795542e-04, -1.112218102384988e-03, -1.169046991267378e-03, -7.322496370200507e-02, -3.314419199420598e-02, -1.902471843781103e-03, -2.070667508672941e-03, -1.902471843781091e-03, -2.070667508672941e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.360087580922066e-09, 1.699680376684791e-10, -9.358181028853170e-09, -9.359981758884938e-09, 1.699677911045265e-10, -9.358104441851266e-09, -9.359324087578305e-09, 1.699547462735486e-10, -9.357250881826714e-09, -9.360507932790816e-09, 1.699582283170237e-10, -9.358762245338196e-09, -9.360038275883405e-09, 1.699581088711227e-10, -9.357919653784204e-09, -9.360038275883405e-09, 1.699581088711227e-10, -9.357919653784204e-09, -2.138128922300026e-05, 2.758858546504131e-06, -2.137982339945743e-05, -2.138416908060473e-05, 2.759473816214609e-06, -2.138169426795299e-05, -2.143849206503228e-05, 2.774025422388848e-06, -2.144850342455879e-05, -2.141649548943290e-05, 2.754809608201874e-06, -2.141571046344994e-05, -2.138648628697892e-05, 2.763369390847232e-06, -2.146333819050379e-05, -2.138648628697892e-05, 2.763369390847232e-06, -2.146333819050379e-05, -5.066466908755691e-03, 2.709323143821118e-03, -5.176895960621028e-03, -5.047076006880409e-03, 2.698168663263616e-03, -5.170643615729822e-03, -4.862739296661272e-03, 2.380933893684406e-03, -4.863045410721725e-03, -5.212892198658536e-03, 2.011915533646660e-03, -5.300042230407614e-03, -5.501699602663182e-03, 2.115030925736313e-03, -5.464036644241723e-03, -5.501699602663182e-03, 2.115030925736313e-03, -5.464036644241723e-03, -2.363140631485781e+00, 1.113778575282640e+00, -1.310362771873759e+00, -2.275284476753163e+00, 1.131585167720763e+00, -1.284623757590014e+00, -9.661425968936118e-04, 1.707854955393021e-03, -1.295721368380512e-03, -4.740062998862886e-01, 7.088238681920542e-01, -2.213552757089188e+00, -1.998383895703846e+00, 1.017636528647748e+00, 7.142311862560760e-01, -1.998383895703845e+00, 1.017636528647750e+00, 7.142311862560768e-01, -1.045427689911739e+02, 2.691692589173089e+03, -8.810341936075096e+03, 1.043135546645721e+02, 8.378791168731273e+02, -1.150999342978984e+03, 5.894982008062108e+00, 5.206546797122851e+00, 3.859303807836241e+00, 8.179086576404113e+02, 3.474425544423646e+02, 8.159562699523372e+02, 2.919340444571888e+02, 2.860361857526898e+02, 1.723489995013629e+03, 2.919340444572016e+02, 2.860361857526883e+02, 1.723489995013630e+03, -2.557471884052870e-06, 2.064120283196137e-07, -2.559290284909815e-06, -2.558284085140114e-06, 2.069595826936506e-07, -2.560066064732370e-06, -2.557267187858782e-06, 2.064098853411341e-07, -2.559135779358334e-06, -2.557965841243511e-06, 2.068941926853079e-07, -2.559855920908981e-06, -2.558081399888545e-06, 2.067032181027890e-07, -2.559702936464961e-06, -2.558081399888545e-06, 2.067032181027890e-07, -2.559702936464961e-06, -1.447713006517155e-04, 1.675754422131500e-05, -1.432499446566491e-04, -1.340555375546439e-04, 1.543449230145285e-05, -1.329931894048089e-04, -1.414717021479674e-04, 1.588213932057392e-05, -1.412490471294327e-04, -1.316571718922486e-04, 1.467182024214476e-05, -1.313047320085101e-04, -1.406245629653106e-04, 1.642715161878144e-05, -1.378462882679779e-04, -1.406245629653106e-04, 1.642715161878144e-05, -1.378462882679779e-04, -1.912999953297612e-02, 1.251669896468412e-02, -1.900182083659287e-02, -1.522652349030727e-02, 1.338874693179927e-02, -1.480027542983357e-02, -3.147012862274270e-02, 2.064571703266417e-02, -2.609139098970614e-02, -3.341338453754290e-02, 2.964080361125809e-02, -2.629338232184677e-02, -1.445024792389026e-02, 1.167016908189115e-02, -2.160315533654572e-02, -1.445024792389027e-02, 1.167016908189114e-02, -2.160315533654573e-02, -8.945572944072810e-01, 1.013006592168991e+00, -5.838841336159084e-01, -2.172868608365305e+00, 9.370545638571657e-01, -2.130482756695715e+00, -7.409950775764118e-03, 1.112408591820349e+00, -7.903766860292677e-01, -2.601837912969889e-04, 1.247282384233750e-04, -2.606747156001924e-04, -4.521519512421317e+00, 6.279807259259585e-01, -7.357566848930853e+00, -4.521519512421373e+00, 6.279807259259206e-01, -7.357566848930857e+00, 1.635790147698657e+03, 5.902777584243285e+02, 1.450212500973293e+03, 9.782901764465403e+02, 3.731293901041719e+02, 9.252390704620443e+02, 4.362015993956442e+03, 1.952914771464913e+03, 4.228178619214376e+03, 1.997936373272892e+00, 2.370384632887535e+00, -3.434714054285100e-01, 2.700059535137150e+03, 8.008422074390305e+02, 2.950719455328264e+02, 2.700059535137145e+03, 8.008422074390303e+02, 2.950719455328115e+02, -1.401701937988743e-02, 1.147494335234051e-02, -1.368523545068198e-02, -1.451106460076996e-02, 1.049780980722926e-02, -1.418932543303361e-02, -1.440799629513238e-02, 1.082048215294510e-02, -1.407879216321790e-02, -1.426093212904896e-02, 1.110366068540403e-02, -1.393036197302689e-02, -1.434167401897776e-02, 1.096074572509396e-02, -1.401096393745859e-02, -1.434167401897776e-02, 1.096074572509396e-02, -1.401096393745859e-02, -1.595257912319586e-02, 1.496753871155219e-02, -1.554755723854057e-02, -2.504499031313259e-02, 1.251231139611195e-02, -2.451853898660932e-02, -2.190581579574326e-02, 1.259085545422972e-02, -2.142228784251285e-02, -1.959168376489696e-02, 1.314825954923481e-02, -1.923902859664534e-02, -2.079997083811999e-02, 1.293317825686115e-02, -2.029631236026523e-02, -2.079997083811999e-02, 1.293317825686115e-02, -2.029631236026523e-02, -1.400790727033776e-02, 1.205622557319765e-02, -1.333327931524149e-02, -7.251688670548584e-01, 3.288432120036838e-01, -7.113361327828152e-01, -4.105466120791515e-01, 2.108522491860608e-01, -3.888895877527404e-01, -1.526372911956951e-01, 1.111199302289075e-01, -1.486688351569267e-01, -2.410499322866778e-01, 1.506759917820290e-01, -2.418614881576081e-01, -2.410499322866780e-01, 1.506759917820289e-01, -2.418614881576082e-01, -4.988038640143858e-02, 2.882362875294387e-02, -4.617058384109349e-02, -8.063120683966243e+00, 9.627091651908412e+00, -1.361619622123158e+00, 2.578456932819634e+00, 2.970313033869382e+00, 2.100132639258672e+00, -1.837330887307267e-01, 1.534267296367740e-01, -1.465211401715559e-01, -1.231394028964740e+00, 2.505158629607186e+00, -3.051608107580353e+00, -1.231394028964741e+00, 2.505158629607169e+00, -3.051608107580352e+00, 1.581523637735721e+02, 6.329641056734451e+01, 1.513575481050793e+02, 1.996208288133101e+04, 8.770284391659925e+03, 2.605939563830916e+04, 4.084256719757768e+03, 2.193811091537201e+03, 3.915775108720729e+03, -2.355853470780243e+00, 4.726101197427621e+00, -2.741183301653015e+01, 4.331793427846835e+03, 1.089913012534649e+03, 5.511794578758661e+02, 4.331793427846846e+03, 1.089913012534657e+03, 5.511794578759323e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan0_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.476755975661047e-04, 7.475465593157115e-04, 7.476738534420100e-04, 7.475452822665467e-04, 7.476486232631673e-04, 7.475147258099070e-04, 7.476679155694977e-04, 7.475390623691364e-04, 7.476758346132774e-04, 7.475143360218648e-04, 7.476758346132774e-04, 7.475143360218648e-04, 6.417030272641602e-03, 6.413638099765386e-03, 6.418234868874314e-03, 6.414130192951239e-03, 6.437198873984752e-03, 6.440909697459170e-03, 6.439376710712790e-03, 6.437133548820428e-03, 6.410745743856437e-03, 6.459070758441131e-03, 6.410745743856437e-03, 6.459070758441131e-03, 1.049276370028407e-02, 1.066562383971267e-02, 1.043997961054841e-02, 1.061640108813714e-02, 9.566036983337257e-03, 9.701669682051247e-03, 1.115443936177168e-02, 1.135385070552083e-02, 1.077917377756410e-02, 1.179438132288560e-02, 1.077917377756410e-02, 1.179438132288560e-02, 7.684205106353151e-02, 4.539807223475049e-02, 7.530258324581075e-02, 4.651898559934339e-02, 1.123110696084897e-03, 1.955595995593000e-03, 1.041704258814859e-02, 4.336763478713374e-02, 9.360633510754016e-02, -1.896358001399442e-02, 9.360633510754013e-02, -1.896358001399453e-02, -1.714071291526311e-03, 1.524516972740772e-02, -3.918714125462361e-04, 2.570503928298781e-03, 4.375217209004327e-04, 1.091014357025899e-03, -1.323316218872228e-07, 3.026328962433467e-06, 1.743984523282776e-04, -4.031627905460035e-05, 1.743984523282642e-04, -4.031627905459911e-05, 2.891075670123073e-03, 2.891161635902778e-03, 2.892648710433005e-03, 2.892657533897572e-03, 2.890955972261559e-03, 2.891055323041878e-03, 2.892295457843258e-03, 2.892440215250620e-03, 2.892048877566286e-03, 2.891927988698006e-03, 2.892048877566286e-03, 2.891927988698006e-03, 8.622037183540102e-03, 8.535932990710064e-03, 8.200971735376639e-03, 8.133867163289512e-03, 8.321648354922259e-03, 8.349402670118126e-03, 7.932035190534682e-03, 7.952047524204668e-03, 8.588266585633782e-03, 8.327829066495538e-03, 8.588266585633782e-03, 8.327829066495538e-03, 2.443317706506040e-02, 2.402136339523756e-02, 2.162799958038774e-02, 2.113395783186697e-02, 2.863123247250240e-02, 2.846416293261212e-02, 2.697723774493113e-02, 2.650964505507452e-02, 2.091556672647994e-02, 2.607955203152462e-02, 2.091556672647996e-02, 2.607955203152462e-02, 8.579371784596614e-03, 6.933322901898669e-03, 7.711248435067940e-02, 7.692811782666223e-02, 3.391986701703001e-03, 7.885424884518853e-03, 1.205931086651109e-02, 1.206760034994911e-02, 4.506332442735829e-02, 8.365412479155090e-02, 4.506332442735662e-02, 8.365412479155143e-02, -4.029859084136135e-09, 5.209628521880695e-08, 4.495096115599893e-07, 1.597171166315069e-08, -2.558194636680923e-07, 5.941828035840798e-07, 7.931101387769285e-04, 4.183203067646211e-03, -3.739314091184118e-06, 4.815949673528319e-04, -3.739314091184071e-06, 4.815949673528366e-04, 1.594373043551138e-02, 1.577335166523682e-02, 1.713110333662597e-02, 1.699204605367262e-02, 1.684381329840333e-02, 1.669132085092342e-02, 1.650039929673821e-02, 1.633821678237867e-02, 1.668540978803499e-02, 1.652698692015633e-02, 1.668540978803498e-02, 1.652698692015633e-02, 1.627219140400986e-02, 1.601280791463852e-02, 2.121066482374839e-02, 2.108327306835223e-02, 2.029261143489981e-02, 2.015760065762539e-02, 1.918052775615316e-02, 1.907155240470555e-02, 1.988984384195087e-02, 1.966091371745974e-02, 1.988984384195088e-02, 1.966091371745975e-02, 2.262236336127901e-02, 2.191229546407305e-02, 5.279737634746280e-02, 5.290063720131766e-02, 4.987325402771037e-02, 4.891295485448838e-02, 3.990828257072761e-02, 3.962130949411185e-02, 4.358296791092366e-02, 4.367365072354813e-02, 4.358296791092366e-02, 4.367365072354813e-02, 3.026829623983024e-02, 2.861704550602457e-02, 3.919248089054249e-03, 1.991067099761836e-03, 8.997212736921623e-04, 1.269927286098450e-03, 3.836982607053972e-02, 3.241647678249385e-02, 6.254641545049896e-03, 1.363933005886119e-02, 6.254641545049924e-03, 1.363933005886121e-02, 2.201446697578214e-06, 1.770863540657783e-06, 5.361901843504675e-09, -9.198210699011337e-09, -8.258501282608611e-07, 5.218830156858298e-07, 6.329869321306900e-03, 7.083487318118580e-02, -7.608107574936477e-06, 4.025391466808198e-04, -7.608107574936019e-06, 4.025391466808038e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05