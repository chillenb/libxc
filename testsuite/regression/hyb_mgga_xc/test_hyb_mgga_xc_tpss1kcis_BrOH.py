
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_tpss1kcis_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.949786912317768e+01, -1.949791699472837e+01, -1.949822396645633e+01, -1.949744269520826e+01, -1.949789353008250e+01, -1.949789353008250e+01, -3.052749125038568e+00, -3.052760036608202e+00, -3.053293062378441e+00, -3.054689882939254e+00, -3.052757699088690e+00, -3.052757699088690e+00, -6.148811744245855e-01, -6.145984262979427e-01, -6.092427240831234e-01, -6.143702204014614e-01, -6.147775879286255e-01, -6.147775879286255e-01, -1.928864125310411e-01, -1.942149871075853e-01, -7.169731598723150e-01, -1.595716186666177e-01, -1.932663907182288e-01, -1.932663907182288e-01, -1.495877901160776e-02, -1.567294180520615e-02, -6.283524702375280e-02, -7.146817163941106e-03, -1.551104568816491e-02, -1.551104568816491e-02, -4.810030586520952e+00, -4.810957435367556e+00, -4.810165691989164e+00, -4.810884344121718e+00, -4.810451199722124e+00, -4.810451199722124e+00, -1.837275535140944e+00, -1.849579396520756e+00, -1.835318755063568e+00, -1.844856608302645e+00, -1.848271814084711e+00, -1.848271814084711e+00, -5.448769108154269e-01, -5.743702620525605e-01, -5.135523351902503e-01, -5.243490345598726e-01, -5.663096377206291e-01, -5.663096377206291e-01, -1.290935582519780e-01, -2.078991933480180e-01, -1.268816433367803e-01, -1.646143326189357e+00, -1.419185948284060e-01, -1.419185948284060e-01, -6.893498636516523e-03, -7.882782718840119e-03, -5.915657047575747e-03, -8.548704207253632e-02, -7.187001759182888e-03, -7.187001759182888e-03, -5.576841481505642e-01, -5.654204272558730e-01, -5.638469614267658e-01, -5.618245798801773e-01, -5.629365383195850e-01, -5.629365383195850e-01, -5.350308784695246e-01, -4.855304395489692e-01, -5.055892956485168e-01, -5.230260404753493e-01, -5.142904225697350e-01, -5.142904225697350e-01, -5.987246920154419e-01, -2.466120474428600e-01, -2.794741416315046e-01, -3.406063958941664e-01, -3.075597707534057e-01, -3.075597707534056e-01, -4.366082636987698e-01, -5.845769083076390e-02, -7.769459126086327e-02, -3.204806377057326e-01, -1.070825473740097e-01, -1.070825473740097e-01, -1.753710174779656e-02, -2.152143946406201e-03, -4.093031634267715e-03, -1.022670655162683e-01, -6.109553802596196e-03, -6.109553802596187e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_tpss1kcis_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.422561933258321e+01, -2.422566679833702e+01, -2.422606046840967e+01, -2.422528883852003e+01, -2.422564272394003e+01, -2.422564272394003e+01, -3.755700913969571e+00, -3.755735462968432e+00, -3.756922828362432e+00, -3.757035981356247e+00, -3.755726266208409e+00, -3.755726266208409e+00, -7.443349447847345e-01, -7.429404674060653e-01, -7.038750402587107e-01, -7.122553141115563e-01, -7.438307258764933e-01, -7.438307258764933e-01, -2.079250356461268e-01, -2.124894853363465e-01, -9.166072925796591e-01, -1.560314051316308e-01, -2.092850918661006e-01, -2.092850918661006e-01, -2.024300540197705e-02, -2.121804829718397e-02, -8.147489976360765e-02, -9.602704650952854e-03, -2.100240173922535e-02, -2.100240173922535e-02, -5.970175883418818e+00, -5.969877236987717e+00, -5.970211482860034e+00, -5.969975807477923e+00, -5.969934698190835e+00, -5.969934698190835e+00, -2.186450772263837e+00, -2.213268924300413e+00, -2.176340661240795e+00, -2.197666379556417e+00, -2.216889984003471e+00, -2.216889984003471e+00, -6.865311813960788e-01, -7.530866345525044e-01, -6.476011198760325e-01, -6.857749876977851e-01, -7.141008901745514e-01, -7.141008901745514e-01, -1.421571250291188e-01, -2.007974048498495e-01, -1.386400796465729e-01, -2.171816737665997e+00, -1.453699517563372e-01, -1.453699517563372e-01, -9.261071006502723e-03, -1.060037754895136e-02, -7.974177265094523e-03, -1.057426188126662e-01, -9.675077591212443e-03, -9.675077591212443e-03, -7.180501219333469e-01, -7.113921560278988e-01, -7.119425315623709e-01, -7.133901607290861e-01, -7.125117950330919e-01, -7.125117950330919e-01, -7.030908852008512e-01, -6.246546234037941e-01, -6.497691624574570e-01, -6.637169361074293e-01, -6.569834442175422e-01, -6.569834442175422e-01, -7.872027783844114e-01, -2.518061312927503e-01, -3.140963311502596e-01, -4.243563543895946e-01, -3.717064448216201e-01, -3.717064448216200e-01, -5.495145749719982e-01, -7.696257315559334e-02, -9.848639727044352e-02, -4.052466376270679e-01, -1.222366986689500e-01, -1.222366986689500e-01, -2.373022070390584e-02, -2.879650897650214e-03, -5.488846440441836e-03, -1.168590918216433e-01, -8.228382780683170e-03, -8.228382780683155e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss1kcis_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.133161761214711e-08, -1.133191427307701e-08, -1.133257935945535e-08, -1.132773624725810e-08, -1.133177955462671e-08, -1.133177955462671e-08, -9.608990475146545e-06, -9.610649597653305e-06, -9.658823243231793e-06, -9.629067027851558e-06, -9.609320175016064e-06, -9.609320175016064e-06, -3.305063236784517e-03, -3.324806867387073e-03, -3.788237618238032e-03, -3.712300893904695e-03, -3.312416040265954e-03, -3.312416040265954e-03, -3.449212630549237e-01, -3.264298760455133e-01, -6.787071809668158e-04, -7.955907754073306e-01, -3.396925942506218e-01, -3.396925942506218e-01, 1.813231470663832e+01, 1.709348602062284e+01, -3.052852834221098e-01, 3.133409131848476e+01, 1.798314156881808e+01, 1.798314156881808e+01, -4.792192488494411e-06, -4.841471106137639e-06, -4.797241429238739e-06, -4.835547627413317e-06, -4.817321739721707e-06, -4.817321739721707e-06, -4.173854271702464e-05, -3.997462388578345e-05, -4.125212814727418e-05, -3.981397470615212e-05, -4.131711875973905e-05, -4.131711875973905e-05, -1.151723343366363e-02, -8.344083967971443e-03, -1.244628276401116e-02, -8.837514554159568e-03, -1.144805912608969e-02, -1.144805912608969e-02, -8.530700708832757e-01, -3.000711536443804e-01, -9.777021813231255e-01, -8.035620328924683e-05, -9.587902732868586e-01, -9.587902732868586e-01, 3.459225315528342e+01, 2.979976419507462e+01, 1.162701867544764e+02, -1.055876223123524e+00, 4.865340272139964e+01, 4.865340272139957e+01, -7.435223758261972e-02, -5.138790948485263e-02, -6.133351282831511e-02, -6.965827601650296e-02, -6.552122576698419e-02, -6.552122576698415e-02, -1.542431437289546e-02, -6.691884781940200e-03, -1.101015511931712e-02, -2.163755325824435e-02, -1.530013544569178e-02, -1.530013544569179e-02, -5.266132407266413e-03, -1.414624843233434e-01, -7.306866767900785e-02, -3.814103953146859e-02, -4.449677909336591e-02, -4.449677909336599e-02, -1.424977143699759e-02, 7.758203622061877e-02, -7.486256834681112e-01, -6.058663422957298e-02, -1.322412800339610e+00, -1.322412800339610e+00, 1.109958603413591e+01, 2.060034377223029e+02, 9.037731226371542e+01, -1.596508494556858e+00, 8.484306795396996e+01, 8.484306795396988e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss1kcis_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss1kcis_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss1kcis", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.594735814157483e-03, 1.594794314951371e-03, 1.595048048902228e-03, 1.594096855769636e-03, 1.594766642443527e-03, 1.594766642443527e-03, 2.264152920986613e-03, 2.264904285189874e-03, 2.288340283809290e-03, 2.290813139971524e-03, 2.264275033546446e-03, 2.264275033546446e-03, 2.842553837140060e-03, 2.847642654656115e-03, 2.763523883601151e-03, 3.075714037367396e-03, 2.845018902867166e-03, 2.845018902867166e-03, -2.311171023309729e-03, -1.784907996474698e-03, 2.880766234473344e-04, -1.166129517307203e-03, -2.125747010995304e-03, -2.125747010995304e-03, -6.520741189924055e-05, -7.098732997259608e-05, -4.170082721819273e-04, -5.331531653124754e-06, -7.270085792444877e-05, -7.270085792444871e-05, 1.085551058598317e-02, 1.097203394748228e-02, 1.086871745892956e-02, 1.095923550954781e-02, 1.091321205823461e-02, 1.091321205823461e-02, 2.052285751228384e-03, 2.096354399701776e-03, 1.922565905757606e-03, 1.951666453987425e-03, 2.261085491651279e-03, 2.261085491651279e-03, 2.004188755994212e-02, 1.379704541968188e-02, 1.519758344994791e-02, 1.016157101918057e-02, 2.378956054215794e-02, 2.378956054215794e-02, -1.120054420692204e-03, -1.937809891002275e-03, -1.259488260172214e-03, 2.445062372736872e-03, -1.913714282871924e-03, -1.913714282871924e-03, -7.566986478517157e-06, -8.175181360366236e-06, -1.802808738620423e-05, -5.204717811536005e-04, -8.356548714874311e-06, -8.356548714874300e-06, 1.292191049012925e-01, 1.282596742674164e-01, 1.424753526029012e-01, 1.485313855916701e-01, 1.464702444356818e-01, 1.464702444356818e-01, 2.326315426591218e-02, 1.161785545714243e-02, 2.398045430620332e-02, 4.667987762341748e-02, 3.334998606974467e-02, 3.334998606974469e-02, 9.881888267820962e-03, -1.798757537247910e-03, 1.015377814126903e-03, 1.250245956333823e-02, 5.593265831097176e-03, 5.593265831097170e-03, 1.268713793949507e-02, -3.365659598555135e-04, -5.178854354552713e-04, 1.865438873549890e-02, -1.316569899087839e-03, -1.316569899087844e-03, -3.492209891369096e-05, -6.797098628154516e-07, -4.815382159821900e-06, -1.422745004086978e-03, -8.482594090532636e-06, -8.482594090532602e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05