
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m08_hx_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.318648389718875e-01, -3.318823999492189e-01, -3.319553945022169e-01, -3.316954176492736e-01, -3.318323826066277e-01, -3.318323826066277e-01, 9.589735161830048e-03, 9.558836236477014e-03, 8.775201357764611e-03, 9.372354793117391e-03, 9.189217568339859e-03, 9.189217568339859e-03, -1.878384603955122e-02, -1.864874041707092e-02, -1.432281351076065e-02, -7.993972987309259e-03, -1.020045693734874e-02, -1.020045693734874e-02, -1.893320520108040e-04, 6.461178192811185e-04, -8.980993702452306e-02, -5.750217120292513e-02, -2.478878161482329e-02, -2.478878161482362e-02, -2.628234402974932e-02, -2.743709573471780e-02, -9.808076707926047e-02, -1.638888520752654e-02, -1.999197963744861e-02, -1.999197963744861e-02, -2.007374173206427e-01, -1.958090301658282e-01, -2.004639772459263e-01, -1.961161835645447e-01, -1.982943536351439e-01, -1.982943536351439e-01, 5.336150042723502e-02, 4.915026710569259e-02, 5.794351493672834e-02, 5.439502850598322e-02, 4.868408535102721e-02, 4.868408535102721e-02, -3.203100516061524e-02, -7.101687482214285e-02, -2.137319814279283e-02, -4.706730633704123e-02, -3.829629579335548e-02, -3.829629579335548e-02, -1.181134448825546e-01, -2.074574048038271e-02, -1.219263941063126e-01, -6.967284425174219e-02, -7.604754899017875e-02, -7.604754899017882e-02, -1.303144792444954e-02, -1.606641904436045e-02, -1.266270134777385e-02, -1.201179238975496e-01, -1.494155491099031e-02, -1.494155491099072e-02, -9.762729307677084e-02, -1.448994118367410e-02, -2.463761676587329e-02, -4.971171200709486e-02, -3.522891184689846e-02, -3.522891184689846e-02, -1.523553283091646e-01, -1.220614013015531e-02, -3.764424126233398e-02, -8.768406893314229e-02, -6.327860952646709e-02, -6.327860952646705e-02, -6.322002851980263e-02, 1.012279538188800e-03, 1.094565441585389e-02, 3.120440587547872e-03, 1.523998546059246e-02, 1.523998546059238e-02, -5.039041864407976e-03, -9.556596536949812e-02, -1.131901290176143e-01, -1.366642448881011e-02, -1.134914553494124e-01, -1.134914553494142e-01, -3.500542074385377e-02, -4.844826747027498e-03, -9.588179151006189e-03, -1.136913553826738e-01, -1.394876284953032e-02, -1.394876284952986e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m08_hx_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.234909484377777e+00, -1.234822119522930e+00, -1.234480763138237e+00, -1.235773304040275e+00, -1.235090077249083e+00, -1.235090077249083e+00, -2.415752055622820e-01, -2.415607327840719e-01, -2.411322497030766e-01, -2.413836195672640e-01, -2.413266830344670e-01, -2.413266830344670e-01, -1.349032722261363e-02, -1.378781917159154e-02, -1.804947758475112e-02, -9.066919721564244e-03, -1.173955902582613e-02, -1.173955902582613e-02, 8.745136456699424e-03, 5.480195930803624e-03, -8.203636484182433e-02, 7.952230741748105e-02, 5.463568771338477e-02, 5.463568771337626e-02, -3.361908879612156e-02, -3.505523947889606e-02, -1.114346372823798e-01, -2.117369695307891e-02, -2.573050707348076e-02, -2.573050707348079e-02, 8.112497320293768e-01, 8.553384001409124e-01, 8.138452302064709e-01, 8.527391878672774e-01, 8.332793676370309e-01, 8.332793676370309e-01, -1.827362190418392e-01, -1.973784911425847e-01, -1.680177567071222e-01, -1.842718143620753e-01, -1.967989970230978e-01, -1.967989970230978e-01, -1.362936058468940e-01, -1.134709419823403e-01, -1.338591590926141e-01, -8.474528960306252e-02, -1.639781204041420e-01, -1.639781204041420e-01, -2.006441258805857e-02, 6.866440485508010e-02, -4.090673784783193e-02, -1.888968706710843e-01, 6.531428591000747e-02, 6.531428591003247e-02, -1.689832438352311e-02, -2.076477138828838e-02, -1.641700380032634e-02, -1.022591104532312e-01, -1.932725481757440e-02, -1.932725481757335e-02, -3.045851322378373e-01, -6.833389083135322e-02, -2.202105542096482e-01, -3.103411998887041e-01, -2.726191379195665e-01, -2.726191379195665e-01, -2.016819418562254e-01, -1.196851536510351e-01, -2.099618350723491e-01, -1.375632711916617e-01, -2.585922004613362e-01, -2.585922004613360e-01, -1.418102281279654e-01, 2.445967147014906e-02, 1.377021929595886e-02, -9.745555624249662e-02, -2.811879480025776e-02, -2.811879480025749e-02, -1.381595056823305e-01, -1.096661093738530e-01, -1.183770132662624e-01, -1.187031641577150e-01, -4.359247196677554e-02, -4.359247196681854e-02, -4.445855128430742e-02, -6.352358076610024e-03, -1.248237856282334e-02, -5.375378488305419e-02, -1.806101223723612e-02, -1.806101223723214e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.501548842524587e-10, -3.501936255844703e-10, -3.503333130876769e-10, -3.497597289607888e-10, -3.500648992671345e-10, -3.500648992671345e-10, 1.897433535905759e-06, 1.897398567346656e-06, 1.895092259306447e-06, 1.882549496200808e-06, 1.890138137956924e-06, 1.890138137956924e-06, 6.436101728949315e-03, 6.406435746049403e-03, 5.474219407408457e-03, 5.229415368730702e-03, 5.356467104868325e-03, 5.356467104868325e-03, 6.416024097602060e-01, 6.671810377800881e-01, 4.128047306485505e-03, 5.093419949592515e-01, 7.106412413377131e-01, 7.106412413377060e-01, -1.088483124085791e-01, -1.280817652963302e-01, -5.217560884354072e-01, -3.872340994843399e-02, -8.074815070677920e-02, -8.074815073390150e-02, -3.534056899834434e-07, -3.280389840690908e-07, -3.519525247696638e-07, -3.295815388870057e-07, -3.409370263513397e-07, -3.409370263513397e-07, 1.337634318510139e-05, 1.271248050361624e-05, 1.322233398551004e-05, 1.262421739610212e-05, 1.307550596724172e-05, 1.307550596724172e-05, 6.977203019209996e-03, -1.574125352857771e-02, 1.225434394180769e-02, 1.308673236206377e-02, 5.234273063638149e-03, 5.234273063638149e-03, -1.762569808453121e-01, 2.766514155276544e-01, -2.883545519738122e-01, 3.836573687036031e-05, 3.754035445061714e-01, 3.754035445061754e-01, -3.787466451274274e-02, -4.761942148961451e-02, -4.756953792394885e-01, -7.742075738691874e-01, -2.071089102013023e-01, -2.071089102987315e-01, 3.429074210587773e-01, 1.941154995759587e-01, 2.825083316562838e-01, 3.450825584427109e-01, 3.169770717070932e-01, 3.169770717070932e-01, 4.376837776576577e-01, 1.176752046369094e-02, 5.764780431975480e-03, -2.417457437647980e-02, -6.410696612228402e-03, -6.410696612228385e-03, -5.755814237101995e-03, 1.786235813102817e-01, 1.480388960843532e-01, 7.812926779587027e-02, 1.195193685575329e-01, 1.195193685575330e-01, 1.936136660273723e-02, -4.348306915009931e-01, -5.586419335685061e-01, 9.128432315082152e-02, -5.458646886562453e-01, -5.458646886562603e-01, -1.347260278259801e-01, -5.174272795977315e-02, -6.502509765301248e-02, -6.042220865341253e-01, -2.578776600252894e-01, -2.578776597629384e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [5.953017052292516e-04, 5.952620313192598e-04, 5.951047998210298e-04, 5.956918317159294e-04, 5.953819715887303e-04, 5.953819715887303e-04, 1.046110980933170e-03, 1.046202178585683e-03, 1.048431877500469e-03, 1.050459514324144e-03, 1.049172713556099e-03, 1.049172713556099e-03, -1.706718375299939e-02, -1.699294965280959e-02, -1.485506043715160e-02, -1.553717319787452e-02, -1.543771945617963e-02, -1.543771945617963e-02, -8.415820960533602e-02, -8.578743607066271e-02, -9.293640442916057e-03, -1.140339136803778e-01, -1.089458452537395e-01, -1.089458452537297e-01, -5.676053052885197e-05, -6.855622179176759e-05, -4.840309494311178e-03, -8.556234944777162e-06, -2.303989300145380e-05, -2.303989300145380e-05, -1.361996511455701e-02, -1.438198570357973e-02, -1.366440853164652e-02, -1.433662480020375e-02, -1.399925955254257e-02, -1.399925955254257e-02, 2.510987381292103e-03, 2.860494177461457e-03, 2.241792464814127e-03, 2.610622987934625e-03, 2.816966218388387e-03, 2.816966218388387e-03, 2.254573325009909e-02, 5.439741910722960e-02, 1.562345376205592e-02, 2.157582727417011e-03, 4.006801503196315e-02, 4.006801503196315e-02, -6.299975660915419e-02, -8.446215255206872e-02, -5.445582959374308e-02, 4.202218049158744e-03, -1.210240985619475e-01, -1.210240985619637e-01, -2.796283236846392e-06, -7.082588972956442e-06, -2.646820220953782e-05, -2.385827376062679e-02, -1.738139545889778e-05, -1.738139545739382e-05, -1.034172769964137e-01, -6.117900341384912e-01, -4.948100270240556e-01, -3.242986924332150e-01, -4.159495871321913e-01, -4.159495871321913e-01, -3.329488115675861e-01, 2.045127266522098e-02, 9.436381932438791e-02, 1.143981097849245e-01, 1.694508120602388e-01, 1.694508120602387e-01, 4.878239558183063e-02, -5.597644531702878e-02, -5.881642656989491e-02, 7.585299205856933e-03, -3.828296424006240e-02, -3.828296424006191e-02, 2.786098876113146e-02, -4.305118121261105e-03, -9.714142679996826e-03, 3.678005751726975e-02, -6.287109143778449e-02, -6.287109143777420e-02, -7.240936826819128e-05, -1.860908578197010e-07, -4.663011923886740e-06, -5.846289289960976e-02, -1.980368194084443e-05, -1.980368196160791e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05