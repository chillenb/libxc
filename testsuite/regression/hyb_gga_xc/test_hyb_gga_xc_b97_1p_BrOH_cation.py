
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_1p_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.789874519931909e+01, -1.789878091387526e+01, -1.789897736711875e+01, -1.789844524281174e+01, -1.789871870510366e+01, -1.789871870510366e+01, -3.004206449849496e+00, -3.004189894542858e+00, -3.003880084613275e+00, -3.004979195979848e+00, -3.004257876494913e+00, -3.004257876494913e+00, -6.258984568761439e-01, -6.257116504113616e-01, -6.232616275288541e-01, -6.274508655210823e-01, -6.301255609148987e-01, -6.301255609148987e-01, -1.979398789063020e-01, -1.989808322986670e-01, -7.206324724095019e-01, -1.694496522538934e-01, -1.825662571399795e-01, -1.825662571399795e-01, -7.128127688878344e-03, -7.568258295566222e-03, -5.145563353440846e-02, -3.857254859687629e-03, -8.594123044867624e-03, -8.594123044867624e-03, -4.394023647199966e+00, -4.394245482368012e+00, -4.394039520917300e+00, -4.394235233338929e+00, -4.394132716045064e+00, -4.394132716045064e+00, -1.817807582247700e+00, -1.826215282246983e+00, -1.818873370899026e+00, -1.826173935836803e+00, -1.822117934528435e+00, -1.822117934528435e+00, -5.366747818623159e-01, -5.721174028113608e-01, -5.006368864999341e-01, -5.112473454053630e-01, -5.439947899187457e-01, -5.439947899187458e-01, -1.323810895891782e-01, -2.143981279560283e-01, -1.237403685358613e-01, -1.643185926011843e+00, -1.448309633775125e-01, -1.448309633775125e-01, -2.924643865147282e-03, -3.762478245097653e-03, -2.844264179525526e-03, -8.465640062033963e-02, -4.511193498979671e-03, -4.511193498979677e-03, -5.270487530885825e-01, -5.285161012183835e-01, -5.286349340487768e-01, -5.283526781031845e-01, -5.285523422440948e-01, -5.285523422440948e-01, -5.099783436460580e-01, -4.656676646466906e-01, -4.777430440654124e-01, -4.916896919814926e-01, -4.842005166447266e-01, -4.842005166447266e-01, -5.985678411049804e-01, -2.530576879491575e-01, -2.833630966495452e-01, -3.381943541973502e-01, -3.075425118956511e-01, -3.075425118956511e-01, -4.312523054167438e-01, -4.885368703141553e-02, -6.844395190600171e-02, -3.213951758310133e-01, -1.048680977133709e-01, -1.048680977133709e-01, -1.039409740584425e-02, -9.221538136634546e-04, -2.046036478768775e-03, -9.932023516721908e-02, -3.982945063939829e-03, -3.982945063939823e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_1p_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.258095012980220e+01, -2.258092453265699e+01, -2.258105837159207e+01, -2.258100313310345e+01, -2.258137132040760e+01, -2.258146070822193e+01, -2.258014990432014e+01, -2.257990344461533e+01, -2.258101799328576e+01, -2.258049989524215e+01, -2.258101799328576e+01, -2.258049989524215e+01, -3.686780173805524e+00, -3.687103281596830e+00, -3.686831710983693e+00, -3.687171742359908e+00, -3.688263458783449e+00, -3.688687994848050e+00, -3.686369216684664e+00, -3.686823259453662e+00, -3.686071280536976e+00, -3.688129114072525e+00, -3.686071280536976e+00, -3.688129114072525e+00, -7.215878845777440e-01, -7.287486580716199e-01, -7.191075231000755e-01, -7.278862946417545e-01, -6.867663478781104e-01, -6.749879114159052e-01, -6.854524963419050e-01, -6.893378287285624e-01, -7.409902443830760e-01, -6.268125286645385e-01, -7.409902443830760e-01, -6.268125286645385e-01, -1.925381079683487e-01, -1.849165082512797e-01, -1.959716126993130e-01, -1.885243996530820e-01, -8.251571636955436e-01, -8.678409033015064e-01, -1.520082779702955e-01, -1.488965652848393e-01, -1.216362729405260e-01, -3.025183169502007e-01, -1.216362729405261e-01, -3.025183169502013e-01, -7.827534558456901e-03, -1.135192219155003e-02, -8.050530117119804e-03, -1.222536823428756e-02, -6.182163712286996e-02, -7.176418927239510e-02, -5.545495294658274e-03, -4.950887462046523e-03, -1.498845165899578e-02, 7.440084587112821e-03, -1.498845165899577e-02, 7.440084587112775e-03, -5.645578986629048e+00, -5.644245441774816e+00, -5.647895124145521e+00, -5.646480718226282e+00, -5.645708539845792e+00, -5.644321502614281e+00, -5.647693988318206e+00, -5.646354430788556e+00, -5.646767238706484e+00, -5.645368525044543e+00, -5.646767238706484e+00, -5.645368525044543e+00, -1.893154120063506e+00, -1.893055705923783e+00, -1.914732934830123e+00, -1.914113962445562e+00, -1.859903228222539e+00, -1.869463537991206e+00, -1.878801428219889e+00, -1.888372493858226e+00, -1.929379396227130e+00, -1.907643513780336e+00, -1.929379396227130e+00, -1.907643513780336e+00, -6.723987348480144e-01, -6.712560862937420e-01, -7.428784254991589e-01, -7.430342522078565e-01, -6.098703380574534e-01, -6.300514532540291e-01, -6.550203262176504e-01, -6.709548933633470e-01, -6.990753295750122e-01, -6.716129478910261e-01, -6.990753295750123e-01, -6.716129478910263e-01, -1.341441202717597e-01, -1.308858326548565e-01, -1.852954266199239e-01, -1.838121383009792e-01, -1.297819985020097e-01, -1.246125226947386e-01, -2.147502597920054e+00, -2.146649967481307e+00, -1.545088507965452e-01, -1.157454381012580e-01, -1.545088507965452e-01, -1.157454381012580e-01, -3.407195820328360e-03, -4.481134131784196e-03, -4.858455561764413e-03, -5.380626382375602e-03, -3.024224015773395e-03, -4.568243395791770e-03, -9.834060943028183e-02, -1.002163689237853e-01, 1.036154977846664e-03, -8.822373875725027e-03, 1.036154977846685e-03, -8.822373875725043e-03, -6.788032574897870e-01, -6.806028254811937e-01, -6.848762107169734e-01, -6.867890791758618e-01, -6.838693100125335e-01, -6.856848654683443e-01, -6.819881322373846e-01, -6.837787945643994e-01, -6.830452085380988e-01, -6.848413654714627e-01, -6.830452085380988e-01, -6.848413654714627e-01, -6.587147452476865e-01, -6.603514394366345e-01, -5.443353281147659e-01, -5.466805134160965e-01, -5.867630374530718e-01, -5.894945803366807e-01, -6.283606326837259e-01, -6.302025928034626e-01, -6.079098675685572e-01, -6.098315805406265e-01, -6.079098675685572e-01, -6.098315805406265e-01, -7.771540267849913e-01, -7.780862115220900e-01, -2.259914586352269e-01, -2.247515361634714e-01, -2.751329841416485e-01, -2.752270087041372e-01, -3.899232719855382e-01, -3.915140654553431e-01, -3.277021089126830e-01, -3.280786536549741e-01, -3.277021089126830e-01, -3.280786536549741e-01, -5.023121150560763e-01, -5.059477727352838e-01, -6.376493942320093e-02, -6.503117285455678e-02, -8.334615937053828e-02, -8.832182620794195e-02, -3.855869808455198e-01, -3.905669010449068e-01, -1.187650915449531e-01, -1.033966301422328e-01, -1.187650915449529e-01, -1.033966301422327e-01, -1.278256062340401e-02, -1.557219690866373e-02, -1.232132511081513e-03, -1.254731653865231e-03, -2.090222945922850e-03, -3.338886378020599e-03, -1.078115993321965e-01, -1.081721892434201e-01, 4.369366725316344e-04, -7.846975638445526e-03, 4.369366725316270e-04, -7.846975638445521e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_1p_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.338594688307586e-09, 0.000000000000000e+00, -6.338621372688878e-09, -6.338440317970638e-09, 0.000000000000000e+00, -6.338510460062570e-09, -6.338032030204429e-09, 0.000000000000000e+00, -6.337895516139464e-09, -6.339757793975315e-09, 0.000000000000000e+00, -6.340113600843565e-09, -6.338483578228905e-09, 0.000000000000000e+00, -6.339299024505735e-09, -6.338483578228905e-09, 0.000000000000000e+00, -6.339299024505735e-09, -9.583275918100421e-06, 0.000000000000000e+00, -9.578198756467880e-06, -9.582630324101247e-06, 0.000000000000000e+00, -9.577270232213507e-06, -9.563912089514529e-06, 0.000000000000000e+00, -9.557710927899683e-06, -9.590207128255888e-06, 0.000000000000000e+00, -9.583524914659341e-06, -9.592099060942054e-06, 0.000000000000000e+00, -9.565685841283905e-06, -9.592099060942054e-06, 0.000000000000000e+00, -9.565685841283905e-06, -7.275240373725226e-03, 0.000000000000000e+00, -6.968771307015172e-03, -7.362020080971610e-03, 0.000000000000000e+00, -6.987589777130597e-03, -8.250439344338400e-03, 0.000000000000000e+00, -8.712452765826344e-03, -8.271137976480967e-03, 0.000000000000000e+00, -8.124861752529924e-03, -5.919706224118981e-03, 0.000000000000000e+00, -1.022694948850963e-02, -5.919706224118981e-03, 0.000000000000000e+00, -1.022694948850963e-02, -1.078022721833371e+00, 0.000000000000000e+00, -1.125101676525132e+00, -1.039768749682076e+00, 0.000000000000000e+00, -1.077316649248992e+00, -4.293283911246430e-03, 0.000000000000000e+00, -3.367491451039352e-03, -2.086740006089289e+00, 0.000000000000000e+00, -2.151099001373546e+00, -1.793716524029104e+00, 0.000000000000000e+00, 1.134044735616481e+01, -1.793716524029103e+00, 0.000000000000000e+00, 1.134044735616485e+01, 7.835515966412868e-01, 0.000000000000000e+00, -7.825034466569913e+00, 1.508131198340851e+00, 0.000000000000000e+00, -8.895455258648703e+00, -2.959349135000243e+00, 0.000000000000000e+00, -6.303955706114603e+00, -4.182630117271907e+00, 0.000000000000000e+00, -1.308777445859588e+00, -1.966550249205864e+01, 0.000000000000000e+00, 1.689014553669354e+02, -1.966550249206651e+01, 0.000000000000000e+00, 1.689014553670894e+02, -1.542738885074747e-06, 0.000000000000000e+00, -1.544464420383411e-06, -1.537437171131971e-06, 0.000000000000000e+00, -1.539362409977125e-06, -1.542448422717430e-06, 0.000000000000000e+00, -1.544305839543542e-06, -1.537918550639090e-06, 0.000000000000000e+00, -1.539650178243270e-06, -1.540006761145111e-06, 0.000000000000000e+00, -1.541903348675674e-06, -1.540006761145111e-06, 0.000000000000000e+00, -1.541903348675674e-06, -1.198482232020387e-04, 0.000000000000000e+00, -1.198725117077766e-04, -1.159140319867327e-04, 0.000000000000000e+00, -1.160485508276360e-04, -1.242576387665041e-04, 0.000000000000000e+00, -1.228937301314548e-04, -1.207431890356305e-04, 0.000000000000000e+00, -1.193765631798076e-04, -1.143079396806629e-04, 0.000000000000000e+00, -1.175606494172249e-04, -1.143079396806629e-04, 0.000000000000000e+00, -1.175606494172249e-04, -9.242509240899682e-03, 0.000000000000000e+00, -9.322288486803013e-03, -6.066490834738858e-03, 0.000000000000000e+00, -6.245096519234063e-03, -1.491016918126515e-02, 0.000000000000000e+00, -1.193582453470246e-02, -8.223155559273509e-03, 0.000000000000000e+00, -6.781664009138284e-03, -7.238259909596121e-03, 0.000000000000000e+00, -9.424422649644448e-03, -7.238259909596143e-03, 0.000000000000000e+00, -9.424422649644444e-03, -3.129995251637716e+00, 0.000000000000000e+00, -3.442445863406328e+00, -9.314419071622465e-01, 0.000000000000000e+00, -9.383563302491214e-01, -2.985943092681542e+00, 0.000000000000000e+00, -4.234332491122957e+00, -7.653970822701439e-05, 0.000000000000000e+00, -7.676675446959264e-05, -2.142170428969617e+00, 0.000000000000000e+00, -4.233027710918664e+00, -2.142170428969617e+00, 0.000000000000000e+00, -4.233027710918664e+00, -1.368077257131080e+00, 0.000000000000000e+00, -4.731462343094156e+00, -2.928797650228328e+00, 0.000000000000000e+00, -3.494448522737507e+00, 2.204011997079954e+01, 0.000000000000000e+00, -5.325818693801179e+01, -5.663010740553197e+00, 0.000000000000000e+00, -5.551674328625073e+00, 1.156855260255047e+02, 0.000000000000000e+00, -5.367852033309139e+01, 1.156855260253930e+02, 0.000000000000000e+00, -5.367852033309742e+01, -3.822691474488626e-02, 0.000000000000000e+00, -3.872429479612819e-02, -9.194043192924242e-03, 0.000000000000000e+00, -9.217812839709788e-03, -1.340562978821819e-02, 0.000000000000000e+00, -1.359260848910482e-02, -2.045605589111304e-02, 0.000000000000000e+00, -2.067746308519834e-02, -1.637506531876491e-02, 0.000000000000000e+00, -1.658846641606812e-02, -1.637506531876491e-02, 0.000000000000000e+00, -1.658846641606812e-02, -8.105030916899690e-02, 0.000000000000000e+00, -8.018859706613368e-02, -2.344716294715245e-02, 0.000000000000000e+00, -2.291374624424774e-02, -1.708473223196445e-02, 0.000000000000000e+00, -1.653609913106290e-02, -1.041942855775326e-02, 0.000000000000000e+00, -1.015123101626923e-02, -1.374081341142788e-02, 0.000000000000000e+00, -1.341928458699469e-02, -1.374081341142788e-02, 0.000000000000000e+00, -1.341928458699469e-02, -4.542861327048493e-03, 0.000000000000000e+00, -4.653967992583553e-03, -4.700215310705692e-01, 0.000000000000000e+00, -4.706769645824186e-01, -2.644238642298435e-01, 0.000000000000000e+00, -2.607629741792185e-01, -9.118447450782545e-02, 0.000000000000000e+00, -8.907350945379928e-02, -1.600859357696125e-01, 0.000000000000000e+00, -1.597499749005297e-01, -1.600859357696128e-01, 0.000000000000000e+00, -1.597499749005298e-01, -3.263445584750420e-02, 0.000000000000000e+00, -3.139448046465886e-02, -4.137307576038782e+00, 0.000000000000000e+00, -4.529516193440591e+00, -3.628614148112830e+00, 0.000000000000000e+00, -5.465076005030391e+00, -9.948456971260081e-02, 0.000000000000000e+00, -9.109141052834201e-02, -3.459533830188449e+00, 0.000000000000000e+00, -7.792272145056173e+00, -3.459533830188469e+00, 0.000000000000000e+00, -7.792272145056198e+00, -1.471935892955238e+00, 0.000000000000000e+00, -5.820036149840263e+00, 1.904570643278387e+01, 0.000000000000000e+00, -5.058130984393282e+01, 9.418248498156624e+00, 0.000000000000000e+00, -1.883582072814703e+01, -5.639418350348720e+00, 0.000000000000000e+00, -6.235924358456844e+00, 1.186962547302077e+02, 0.000000000000000e+00, -4.398331859233019e+01, 1.186962547302617e+02, 0.000000000000000e+00, -4.398331859231013e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05