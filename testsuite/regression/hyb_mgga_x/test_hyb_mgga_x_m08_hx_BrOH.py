
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m08_hx_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.952959789495251e+00, -7.952924676937457e+00, -7.952767007004906e+00, -7.953334881798665e+00, -7.952941372613644e+00, -7.952941372613644e+00, -1.897866691072816e+00, -1.897840699735833e+00, -1.897192588862345e+00, -1.899324054284785e+00, -1.897857422386692e+00, -1.897857422386692e+00, -3.630185033037428e-01, -3.634465492618631e-01, -3.753676889679123e-01, -3.793580302586770e-01, -3.631732796862312e-01, -3.631732796862312e-01, -1.243964406758415e-01, -1.249578205270342e-01, -3.764912242228470e-01, -8.030041335928627e-02, -1.245522244779209e-01, -1.245522244779209e-01, 6.573010878731510e-03, 6.806062737100833e-03, 1.519060757515543e-03, 3.490294427309810e-03, 6.734159035515116e-03, 6.734159035515318e-03, -1.785325485807697e+00, -1.784261561391767e+00, -1.785202662565565e+00, -1.784376452102112e+00, -1.784778820533484e+00, -1.784778820533484e+00, -1.238477317855754e+00, -1.248571314675442e+00, -1.240081593205203e+00, -1.248455402171568e+00, -1.242705615868612e+00, -1.242705615868612e+00, -2.774283031057778e-01, -2.491025323738828e-01, -2.912826707815858e-01, -2.481042658499074e-01, -2.614372768680794e-01, -2.614372768680794e-01, -5.080389464238089e-02, -1.249078930305839e-01, -5.108539255025771e-02, -8.118493359333114e-01, -6.727415508149377e-02, -6.727415508149377e-02, 3.360562053927226e-03, 3.825916764852343e-03, 2.861933117552236e-03, -1.278512909530602e-02, 3.491030574160902e-03, 3.491030574160979e-03, -1.470311196688301e-01, -2.152080574286082e-01, -2.109443009194753e-01, -1.951638589047091e-01, -2.048469676343056e-01, -2.048469676343056e-01, -1.277910419041817e-01, -2.635920339247334e-01, -2.179628505601593e-01, -1.988789340208288e-01, -2.070339638446918e-01, -2.070339638446916e-01, -2.701094109297950e-01, -1.608039979365361e-01, -1.809753129230268e-01, -2.080204929052449e-01, -1.939286116361996e-01, -1.939286116361991e-01, -2.665495538129529e-01, 5.085400526255319e-03, -7.991261539865719e-03, -1.877654188582849e-01, -3.929777576518281e-02, -3.929777576518120e-02, 7.798017024304025e-03, 1.069106590191506e-03, 2.017195512287911e-03, -3.729112638559664e-02, 2.975667141933841e-03, 2.975667141934022e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m08_hx_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.862247802656099e+00, -7.862275317262990e+00, -7.862378645124575e+00, -7.861925231924097e+00, -7.862262497150980e+00, -7.862262497150980e+00, -1.892787667167352e+00, -1.892036867290711e+00, -1.869072784702465e+00, -1.875542043702723e+00, -1.892605121809787e+00, -1.892605121809787e+00, -3.115604560403159e-01, -3.114485194496850e-01, -3.267143255691463e-01, -3.202143306501669e-01, -3.115114949049147e-01, -3.115114949049147e-01, -1.211979230691453e-01, -1.169639154985830e-01, -4.497907203214512e-01, -1.307335531138218e-01, -1.198386270778709e-01, -1.198386270778709e-01, 7.327493104872201e-03, 7.436707810939485e-03, -3.401235893689689e-02, 4.547232415617995e-03, 7.356565382131135e-03, 7.356565382134193e-03, -2.056914727207422e+00, -2.071410609496472e+00, -2.058613289788848e+00, -2.069870679421946e+00, -2.063924484205299e+00, -2.063924484205299e+00, -1.251687062971109e+00, -1.226370374255174e+00, -1.260904008042571e+00, -1.246820634409934e+00, -1.215448768154538e+00, -1.215448768154538e+00, -9.442388937112435e-02, -2.527291619799527e-01, -1.822560337222175e-01, -1.958492378742878e-01, -1.430114490128168e-01, -1.430114490128168e-01, -8.583048834311134e-02, -1.787395292921607e-01, -8.510174746941998e-02, -7.759084448821876e-01, -1.066476776896962e-01, -1.066476776896962e-01, 4.367708307107909e-03, 4.944640930123471e-03, 3.689811885539533e-03, -5.952486534377341e-02, 4.520213510355939e-03, 4.520213510355251e-03, 5.167496207396453e-02, -2.591097181903572e-01, -1.653575922745640e-01, -6.542085863149236e-02, -1.171117073679709e-01, -1.171117073679709e-01, -1.090258098155841e-03, -2.761165625955851e-02, -1.247733555531026e-01, -1.929331763088129e-01, -1.926330470046299e-01, -1.926330470046284e-01, -3.029641776191174e-01, -1.761977427659914e-01, -1.585956537369253e-01, -2.207753478691148e-01, -1.946142730443561e-01, -1.946142730443563e-01, -1.852709648153232e-01, -2.485447917387560e-02, -5.494397794344059e-02, -1.574827065375131e-01, -6.979905086817242e-02, -6.979905086817074e-02, 8.842273697884612e-03, 1.421863443097370e-03, 2.658251334144428e-03, -6.670733040672579e-02, 3.869057481110069e-03, 3.869057481117577e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_hx_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.914598247135190e-09, 3.914721569502880e-09, 3.915178588581617e-09, 3.913171440989790e-09, 3.914663898170496e-09, 3.914663898170496e-09, -1.216646535550776e-05, -1.216531010359846e-05, -1.212650294551129e-05, -1.211589891830485e-05, -1.216605530175300e-05, -1.216605530175300e-05, -7.600736133933189e-03, -7.582102802219905e-03, -6.922414995082768e-03, -6.774738915594602e-03, -7.594202877815800e-03, -7.594202877815800e-03, -5.013481109466724e-01, -5.230677790591517e-01, -4.494253465944596e-03, -8.726487062874842e-02, -5.086683848151683e-01, -5.086683848151683e-01, 1.186913409981979e+01, 1.179949746560063e+01, 9.732558501446581e-01, 9.079528254559364e+00, 1.225585576654190e+01, 1.225585576654192e+01, -1.857965264145931e-07, -2.676436435495619e-07, -1.946996840737196e-07, -2.582874928621234e-07, -2.264533008571303e-07, -2.264533008571303e-07, -8.481120028619430e-05, -8.326877871871403e-05, -8.398836143711861e-05, -8.289678185200768e-05, -8.463096222775445e-05, -8.463096222775445e-05, -8.381333141515186e-03, 6.509935526222170e-03, -1.677495366177595e-02, -4.633573726053769e-03, -2.667352169662489e-03, -2.667352169662489e-03, -1.020841065624828e+00, -1.343695114095342e-01, -1.054218525909110e+00, -2.046294523828993e-05, -2.963352117713807e-01, -2.963352117713807e-01, 9.614999838951576e+00, 9.610063163910905e+00, 2.745518847428379e+01, -1.131408181332019e+00, 1.421262329670943e+01, 1.421262329670936e+01, -3.798224344362465e-01, -1.352120011337024e-01, -1.988041384818206e-01, -2.644258969312734e-01, -2.295047784766599e-01, -2.295047784766599e-01, -3.079027645012382e-01, -1.533293558788103e-02, -1.310508190014836e-04, 9.985920793864587e-03, 7.257871691612256e-03, 7.257871691612259e-03, 2.664810291823905e-03, -1.445744687044369e-01, -1.364975345191856e-01, -9.972371643041934e-02, -1.254413450091151e-01, -1.254413450091150e-01, -3.401448753362979e-02, 1.399163941636871e+00, -4.809291272111834e-01, -1.333992921702174e-01, -2.322739346977631e+00, -2.322739346977619e+00, 9.033013271958767e+00, 1.677740021960404e+01, 1.437161682454125e+01, -2.787284804561953e+00, 2.080100899576778e+01, 2.080100899576780e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_hx_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_hx_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.513945524344597e-03, -2.513973689506533e-03, -2.514115072707065e-03, -2.513659240457981e-03, -2.513960177585382e-03, -2.513960177585382e-03, 9.220718973273429e-04, 9.154637039901262e-04, 7.126414572954384e-04, 7.788561044463090e-04, 9.203157937129655e-04, 9.203157937129655e-04, 4.767882288310384e-03, 4.771125734330778e-03, 5.910312596364663e-03, 5.193147117865594e-03, 4.768633537803906e-03, 4.768633537803906e-03, 2.083344029991293e-02, 1.894733913298317e-02, 7.850441812750176e-03, 1.756642790716770e-02, 2.024403193970400e-02, 2.024403193970400e-02, 1.374065681016481e-03, 1.561407632451449e-03, 2.403591216712189e-02, 5.548308265712744e-05, 1.582560883236874e-03, 1.582560883230192e-03, -3.999741111053348e-03, -3.512072009881822e-03, -3.944978627016713e-03, -3.566158596984229e-03, -3.760561748218581e-03, -3.760561748218581e-03, 6.428770692896105e-03, 5.749240052330099e-03, 6.594390642329573e-03, 6.198343896483519e-03, 5.568994263869735e-03, 5.568994263869735e-03, -9.203259056432476e-02, -6.421698717757707e-02, -4.056150919666642e-02, -6.191843214419143e-02, -8.852967205882999e-02, -8.852967205882999e-02, 2.995940084575608e-02, 2.681384048632002e-02, 3.001181215616342e-02, -1.399970993929883e-02, 2.152898946825697e-02, 2.152898946825697e-02, 7.591853204931938e-05, 9.366014356900397e-05, 1.549380577261770e-04, 3.054508034258775e-02, 8.732317267878011e-05, 8.732317267839673e-05, -1.057916688333099e-01, 4.790548376328789e-01, 4.188756835517782e-01, 2.484716155264040e-01, 3.480174963875332e-01, 3.480174963875332e-01, -2.371970581146243e-01, -1.229471981716257e-01, -1.153701185707499e-01, -1.073177843388057e-01, -9.304016864060986e-02, -9.304016864061153e-02, -3.310144983261891e-02, 2.017782817782538e-02, 1.333782507736885e-02, 3.646322022829365e-02, 3.528488894096016e-02, 3.528488894096076e-02, -2.036439931360356e-02, 1.968060120801213e-02, 3.065969139837894e-02, -8.508550463933891e-03, 3.784873412894604e-02, 3.784873412891986e-02, 8.649335054050791e-04, 2.140734158466484e-06, 2.878254920648758e-05, 3.946811566975439e-02, 7.542687968501543e-05, 7.542687968613106e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05