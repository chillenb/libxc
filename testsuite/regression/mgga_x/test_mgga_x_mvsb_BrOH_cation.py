
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mvsb_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.349483645431435e+01, -2.349488284783718e+01, -2.349499120487554e+01, -2.349429459476885e+01, -2.349467360061038e+01, -2.349467360061038e+01, -3.085459573177158e+00, -3.085375837283434e+00, -3.083294992936834e+00, -3.085253224216494e+00, -3.084359496987303e+00, -3.084359496987303e+00, -6.241550869474598e-01, -6.238693338694805e-01, -6.178751810619877e-01, -6.239465497385445e-01, -6.258495279414986e-01, -6.258495279414986e-01, -1.831194612138136e-01, -1.859224557713156e-01, -7.246820591871682e-01, -1.293336172571835e-01, -1.719619933892971e-01, -1.719619933892971e-01, -1.466524898597549e-03, -1.594423020109399e-03, -1.909528563254714e-02, -6.254293414639622e-04, -1.074121958879962e-03, -1.074121958879962e-03, -5.674710156286677e+00, -5.675716253300159e+00, -5.674703617446571e+00, -5.675594500936747e+00, -5.675247059245084e+00, -5.675247059245084e+00, -2.049814366324004e+00, -2.070762711969809e+00, -2.051065677640897e+00, -2.070507448254579e+00, -2.059513539237058e+00, -2.059513539237058e+00, -6.148694658655581e-01, -7.500553679776628e-01, -4.744369344237358e-01, -5.847424692027128e-01, -6.487034242400009e-01, -6.487034242400009e-01, -7.776766621644485e-02, -1.783363830669475e-01, -7.005885879743212e-02, -2.027218335422192e+00, -1.015965909240488e-01, -1.015965909240488e-01, -4.255213443221143e-04, -6.078950114973747e-04, -5.195537522030788e-04, -4.016565541812588e-02, -6.693546603299934e-04, -6.693546603299936e-04, -6.466461020347671e-01, -6.433217729678440e-01, -6.444834176527724e-01, -6.454474906170795e-01, -6.449666244794271e-01, -6.449666244794271e-01, -6.336893941151484e-01, -4.911583183246410e-01, -5.887923646003221e-01, -5.999101854240413e-01, -5.922838832497217e-01, -5.922838832497217e-01, -7.529027006091719e-01, -2.312975596027901e-01, -2.779774124145728e-01, -3.343786423860862e-01, -3.150931463835607e-01, -3.150931463835608e-01, -4.301870540419445e-01, -1.773846316364606e-02, -2.812480898431263e-02, -3.018019576813223e-01, -5.925339313486958e-02, -5.925339313486958e-02, -2.319525178775524e-03, -9.583281635920627e-05, -2.916389720687012e-04, -5.412979336585180e-02, -6.029548254839825e-04, -6.029548254839819e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mvsb_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.407754874241407e+01, -3.407755517579870e+01, -3.407745720440002e+01, -3.407748876813005e+01, -3.407746542006211e+01, -3.407741574277946e+01, -3.407849271196808e+01, -3.407872783283274e+01, -3.407747920291181e+01, -3.407842418733552e+01, -3.407747920291181e+01, -3.407842418733552e+01, -3.357382828787412e+00, -3.358599130228201e+00, -3.357051954849323e+00, -3.358477008903673e+00, -3.351697000635645e+00, -3.351045220387690e+00, -3.349360258649904e+00, -3.350399776541448e+00, -3.357478511724813e+00, -3.345523650492587e+00, -3.357478511724813e+00, -3.345523650492587e+00, -7.461376541330387e-01, -7.482301592963130e-01, -7.450614763107486e-01, -7.479615844366683e-01, -7.320725069420588e-01, -7.245502975215844e-01, -7.236941293116367e-01, -7.238749084662088e-01, -7.557580313970542e-01, -6.980813756547972e-01, -7.557580313970542e-01, -6.980813756547972e-01, -2.301890994917483e-01, -2.338982948018733e-01, -2.314598687061491e-01, -2.361956925867036e-01, -9.145926474858987e-01, -9.350212838273373e-01, -1.944847879575155e-01, -1.980413343485565e-01, -2.397951736621503e-01, -1.432831752375639e-01, -2.397951736621502e-01, -1.432831752375639e-01, -2.587837590601758e-03, -2.830235708729880e-03, -2.795241653182548e-03, -3.096148538407576e-03, -3.432080048046312e-02, -3.728572368914139e-02, -1.195759803503218e-03, -1.161935074997212e-03, -2.182180661376794e-03, -1.050886520937535e-03, -2.182180661376795e-03, -1.050886520937535e-03, -7.994277939405986e+00, -7.992553548964592e+00, -7.990484829032653e+00, -7.988885943503671e+00, -7.994087281562341e+00, -7.992441929955973e+00, -7.990831985902640e+00, -7.989109720264295e+00, -7.992310985110203e+00, -7.990701946323720e+00, -7.992310985110203e+00, -7.990701946323720e+00, -2.074528630367301e+00, -2.080912037255672e+00, -2.073819304511866e+00, -2.078444308503568e+00, -2.170418575279844e+00, -2.151206440981692e+00, -2.165632137308708e+00, -2.147418886882896e+00, -1.950503822689386e+00, -2.067242797162468e+00, -1.950503822689386e+00, -2.067242797162468e+00, -1.063245322775536e+00, -1.075855415801038e+00, -1.136841407188463e+00, -1.080012009685028e+00, -5.023392675776698e-01, -4.761721381182403e-01, -8.701452361572818e-01, -9.436534686373336e-01, -1.216771736996972e+00, -1.047854791191091e+00, -1.216771736996973e+00, -1.047854791191091e+00, -1.376182864172384e-01, -1.393527939165913e-01, -2.486294454473400e-01, -2.496199578031264e-01, -1.218126254423116e-01, -1.302712511104810e-01, -3.203212088836843e+00, -3.202524732115700e+00, -1.610361071143006e-01, -1.687426579420809e-01, -1.610361071143006e-01, -1.687426579420809e-01, -7.911555865106157e-04, -8.265514206079631e-04, -1.151672471674387e-03, -1.164577214617482e-03, -9.288258378156631e-04, -1.025844614309062e-03, -7.480286925464892e-02, -7.529297771699602e-02, -8.387876006513396e-04, -1.408162439722582e-03, -8.387876006513397e-04, -1.408162439722582e-03, -8.581190663865444e-01, -8.613467300496082e-01, -8.641134987209375e-01, -8.671846021515349e-01, -8.617342826131537e-01, -8.648561332168848e-01, -8.599770750248844e-01, -8.631679270155055e-01, -8.608266375713081e-01, -8.639863876927666e-01, -8.608266375713081e-01, -8.639863876927666e-01, -8.287488591566501e-01, -8.317933387997700e-01, 1.305332257353140e+00, 1.756422904634290e+00, -8.997159965311639e-01, -9.033148399689082e-01, -8.294604369608940e-01, -8.317597807527938e-01, -8.497806955496999e-01, -8.531525806284781e-01, -8.497806955496999e-01, -8.531525806284779e-01, -1.259694470826750e+00, -1.282764786813173e+00, -2.958578281072405e-01, -2.970816479620776e-01, -3.378757736733650e-01, -3.397507382040242e-01, -2.650839618824053e-01, -2.694217031133243e-01, -3.532365100978213e-01, -3.517643196400265e-01, -3.532365100978214e-01, -3.517643196400265e-01, -1.802849694944240e-01, -1.443649868258204e-01, -3.294502441030665e-02, -3.330125558878953e-02, -5.146354134960458e-02, -5.422373003655279e-02, -2.810041484363721e-01, -2.293791892799271e-01, -1.021317857542644e-01, -1.113428612590952e-01, -1.021317857542644e-01, -1.113428612590952e-01, -4.293765267153592e-03, -4.535894583987688e-03, -1.773867477909111e-04, -1.896597165661889e-04, -5.162864824044971e-04, -5.707859474346128e-04, -9.728265697221419e-02, -9.916704338278472e-02, -8.595228541723105e-04, -1.241466054880943e-03, -8.595228541723092e-04, -1.241466054880942e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsb_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.767780208820635e-08, 0.000000000000000e+00, 4.764587483704315e-08, 4.767502345963737e-08, 0.000000000000000e+00, 4.764386387975760e-08, 4.766433142943184e-08, 0.000000000000000e+00, 4.762907086170381e-08, 4.769544703051786e-08, 0.000000000000000e+00, 4.766875194649310e-08, 4.767611314652750e-08, 0.000000000000000e+00, 4.765157958223486e-08, 4.767611314652750e-08, 0.000000000000000e+00, 4.765157958223486e-08, -1.129918537303586e-05, 0.000000000000000e+00, -1.127500211693672e-05, -1.129709519387737e-05, 0.000000000000000e+00, -1.127111712182164e-05, -1.122535921280582e-05, 0.000000000000000e+00, -1.120443399462674e-05, -1.136988302887679e-05, 0.000000000000000e+00, -1.134247009948960e-05, -1.131383087448595e-05, 0.000000000000000e+00, -1.127717457940202e-05, -1.131383087448595e-05, 0.000000000000000e+00, -1.127717457940202e-05, -6.608615739876279e-03, 0.000000000000000e+00, -6.411869918723643e-03, -6.697863881350126e-03, 0.000000000000000e+00, -6.430428716061099e-03, -7.740618966013825e-03, 0.000000000000000e+00, -8.466249089054559e-03, -8.899619231078796e-03, 0.000000000000000e+00, -8.882028313423473e-03, -5.846838170079113e-03, 0.000000000000000e+00, -1.208194760510263e-02, -5.846838170079113e-03, 0.000000000000000e+00, -1.208194760510263e-02, -7.512558456134723e-01, 0.000000000000000e+00, -9.864026631010702e-01, -7.615007164394441e-01, 0.000000000000000e+00, -1.002942601071497e+00, -1.428411292395303e-03, 0.000000000000000e+00, -1.484022279243446e-03, 9.002023598194715e-02, 0.000000000000000e+00, -2.650802598778236e-02, -6.283694671334121e-01, 0.000000000000000e+00, 1.312005869876149e+00, -6.283694671334118e-01, 0.000000000000000e+00, 1.312005869876148e+00, 1.286757792640406e+02, 0.000000000000000e+00, 1.141025750273560e+02, 1.271117026461922e+02, 0.000000000000000e+00, 1.111767012236750e+02, 8.580469635026075e+00, 0.000000000000000e+00, 7.728815146238139e+00, 3.333874892963898e+02, 0.000000000000000e+00, 3.266041892095192e+02, 1.639892299395436e+02, 0.000000000000000e+00, 8.961935099294628e+02, 1.639892299395443e+02, 0.000000000000000e+00, 8.961935099294642e+02, 9.315552659811747e-06, 0.000000000000000e+00, 9.321216250853046e-06, 9.268787327975524e-06, 0.000000000000000e+00, 9.276001598073735e-06, 9.311139783244868e-06, 0.000000000000000e+00, 9.318354799435052e-06, 9.271093597562670e-06, 0.000000000000000e+00, 9.277356587979862e-06, 9.292889356259969e-06, 0.000000000000000e+00, 9.298672019037819e-06, 9.292889356259969e-06, 0.000000000000000e+00, 9.298672019037819e-06, -3.173369257288424e-04, 0.000000000000000e+00, -3.173458333926643e-04, -3.309425786925906e-04, 0.000000000000000e+00, -3.315910933148658e-04, -2.824501753088889e-04, 0.000000000000000e+00, -2.915703454190421e-04, -2.983581820094712e-04, 0.000000000000000e+00, -3.072911282540262e-04, -3.700839278244825e-04, 0.000000000000000e+00, -3.292589200266337e-04, -3.700839278244825e-04, 0.000000000000000e+00, -3.292589200266337e-04, -1.898769257860457e-02, 0.000000000000000e+00, -2.528919795356135e-02, -4.196454615638914e-02, 0.000000000000000e+00, -2.549226852335153e-02, -2.172109171927086e-02, 0.000000000000000e+00, -1.277083258037684e-02, -2.647920117480064e-03, 0.000000000000000e+00, -1.210285250410715e-02, -8.066534897637347e-02, 0.000000000000000e+00, -2.492337775837003e-02, -8.066534897637372e-02, 0.000000000000000e+00, -2.492337775837004e-02, 1.419847413324441e+00, 0.000000000000000e+00, 1.421183719465210e+00, -2.727453411747309e-01, 0.000000000000000e+00, -2.687160330237719e-01, 1.980074259984963e+00, 0.000000000000000e+00, 1.666813636851466e+00, -1.356949133671810e-04, 0.000000000000000e+00, -1.364707522053065e-04, 7.718267206889031e-01, 0.000000000000000e+00, 2.622248285172296e-01, 7.718267206889028e-01, 0.000000000000000e+00, 2.622248285172277e-01, 6.792705257899512e+02, 0.000000000000000e+00, 5.957177110349362e+02, 4.415237417539362e+02, 0.000000000000000e+00, 4.137279271072130e+02, 1.834824517990976e+03, 0.000000000000000e+00, 1.815792065009654e+03, 4.859567422466180e+00, 0.000000000000000e+00, 4.322675956951088e+00, 1.122173986026474e+03, 0.000000000000000e+00, 5.744113637714667e+02, 1.122173986026472e+03, 0.000000000000000e+00, 5.744113637714652e+02, 4.445922387246373e-02, 0.000000000000000e+00, 4.355128254781315e-02, 5.093629485764762e-02, 0.000000000000000e+00, 4.978255523802762e-02, 4.848548706014003e-02, 0.000000000000000e+00, 4.741097454001039e-02, 4.658988628297845e-02, 0.000000000000000e+00, 4.562055153100553e-02, 4.752434534683252e-02, 0.000000000000000e+00, 4.650334076893849e-02, 4.752434534683252e-02, 0.000000000000000e+00, 4.650334076893851e-02, 5.432397519905610e-02, 0.000000000000000e+00, 5.311604559962863e-02, -9.908856175730548e-01, 0.000000000000000e+00, -1.123776077301422e+00, 3.784817218018088e-01, 0.000000000000000e+00, 3.504495965731975e-01, 1.236417456170095e-01, 0.000000000000000e+00, 1.206663743184052e-01, 1.998613014681366e-01, 0.000000000000000e+00, 1.908941847211093e-01, 1.998613014681367e-01, 0.000000000000000e+00, 1.908941847211094e-01, -3.656926711843332e-02, 0.000000000000000e+00, -4.393331908687983e-02, -3.066138254749479e-01, 0.000000000000000e+00, -3.056760080400853e-01, -2.580472473281666e-01, 0.000000000000000e+00, -2.623528239229589e-01, -4.119452444208976e-01, 0.000000000000000e+00, -3.919855999432971e-01, -3.088643030701436e-01, 0.000000000000000e+00, -3.154991182561717e-01, -3.088643030701435e-01, 0.000000000000000e+00, -3.154991182561720e-01, -2.412653742510592e-01, 0.000000000000000e+00, -2.504436501301648e-01, 7.246517723503126e+00, 0.000000000000000e+00, 7.372159016312537e+00, 5.497594811316114e+00, 0.000000000000000e+00, 5.273519643180254e+00, -2.838519262694070e-01, 0.000000000000000e+00, -3.034553153177814e-01, 3.089357608295630e+00, 0.000000000000000e+00, 2.896563252443167e+00, 3.089357608295629e+00, 0.000000000000000e+00, 2.896563252443167e+00, 9.280831890742995e+01, 0.000000000000000e+00, 8.951848623914027e+01, 6.373331544737810e+03, 0.000000000000000e+00, 1.075356921336850e+04, 1.437452413628852e+03, 0.000000000000000e+00, 1.370190575520993e+03, 3.421329856598263e+00, 0.000000000000000e+00, 2.795187567503327e+00, 2.026210877682151e+03, 0.000000000000000e+00, 6.712865318070570e+02, 2.026210877682157e+03, 0.000000000000000e+00, 6.712865318070610e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsb_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsb_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.907561431260449e-03, -3.903995318751016e-03, -3.907349799165459e-03, -3.903842386617376e-03, -3.906428225724434e-03, -3.902592855392071e-03, -3.908789790963859e-03, -3.905605096323780e-03, -3.907433918050285e-03, -3.904223146584230e-03, -3.907433918050285e-03, -3.904223146584230e-03, -3.033310107741694e-03, -3.031254089070010e-03, -3.036734722159179e-03, -3.033548104373510e-03, -3.104806364509733e-03, -3.117259063779966e-03, -3.076610675661730e-03, -3.077339133420093e-03, -3.026103377389777e-03, -3.139620516395408e-03, -3.026103377389777e-03, -3.139620516395408e-03, 8.185288754898458e-04, 3.753127712591856e-04, 9.730215462555211e-04, 4.260493308134051e-04, 2.939507123768305e-03, 3.922859944753390e-03, 4.128892708953066e-03, 3.919422456151097e-03, -2.791540112012059e-04, 1.037027286488232e-02, -2.791540112012059e-04, 1.037027286488232e-02, 3.587181326955053e-02, 5.082287457953683e-02, 3.572745488303895e-02, 5.216356293998073e-02, 8.771079868476139e-05, -2.297803772074513e-04, 1.253671801765451e-02, 1.592974274247156e-02, 3.316741492446369e-02, 6.915612863235073e-03, 3.316741492446369e-02, 6.915612863235076e-03, 2.721951347166868e-04, 3.039533875595682e-04, 2.931008114312904e-04, 3.345689989216551e-04, 1.979250501841066e-03, 2.292701990692115e-03, 8.269526853853067e-05, 8.393688089958489e-05, 2.096552596649589e-04, 9.200551122989705e-05, 2.096552596649583e-04, 9.200551122989705e-05, -1.044067735286882e-02, -1.043747024327772e-02, -1.039362619926977e-02, -1.039203178012261e-02, -1.043524294533671e-02, -1.043387738015063e-02, -1.039499867678621e-02, -1.039271997047404e-02, -1.041862241924500e-02, -1.041493732273200e-02, -1.041862241924500e-02, -1.041493732273200e-02, 1.369282409828903e-02, 1.382653896256930e-02, 1.503343784540809e-02, 1.518219471529001e-02, 1.299599307306225e-02, 1.329078850191575e-02, 1.438515101272125e-02, 1.470533151385050e-02, 1.552159906635229e-02, 1.462053924862217e-02, 1.552159906635229e-02, 1.462053924862217e-02, 1.347821826615018e-01, 1.520271324862964e-01, 1.394699695208832e-01, 8.235863229567171e-02, -2.258652124713416e-02, -4.564775278818046e-02, 5.233391518312700e-02, 7.861255195802959e-02, 2.978598475103199e-01, 1.421261876593802e-01, 2.978598475103207e-01, 1.421261876593803e-01, 7.526287520532634e-03, 7.456350673920338e-03, 2.306944552583062e-02, 2.284396617777296e-02, 6.204717954337419e-03, 7.270016783358856e-03, 2.623772474504433e-02, 2.629680636286188e-02, 1.188534458406743e-02, 1.704957002736425e-02, 1.188534458406743e-02, 1.704957002736427e-02, 4.029292117917983e-05, 3.947084102559158e-05, 6.520200501650315e-05, 6.088531976760075e-05, 1.666139598970603e-04, 1.903709593910455e-04, 3.985957448713241e-03, 4.844251989212522e-03, 6.312831358366767e-05, 2.466295189003547e-04, 6.312831358366767e-05, 2.466295189003549e-04, -7.501596776269367e-02, -7.424533385145002e-02, -8.471211847848245e-02, -8.364837664310640e-02, -8.103823113996046e-02, -8.006287297589787e-02, -7.819762901970823e-02, -7.736343547179877e-02, -7.960003779546201e-02, -7.869565484903651e-02, -7.960003779546201e-02, -7.869565484903651e-02, -8.816288144164137e-02, -8.678523738997558e-02, 4.439285671242639e-01, 4.299199571688612e-01, -5.199519332376532e-01, -4.803196745260359e-01, -1.727762818759434e-01, -1.701042302407725e-01, -2.738177164893933e-01, -2.621446024125291e-01, -2.738177164893934e-01, -2.621446024125293e-01, 1.737894979336109e-01, 1.942293012516680e-01, 3.238289849151534e-02, 3.252123444443548e-02, 3.561621323143167e-02, 3.676915157218489e-02, 3.798099141408207e-02, 3.474817851456288e-02, 4.893641976380211e-02, 4.961724476753650e-02, 4.893641976380211e-02, 4.961724476753653e-02, 3.661937388045215e-02, 3.120056616476110e-02, 2.294732718445170e-03, 2.234866052295438e-03, 2.742922008249511e-03, 2.982623795104663e-03, -1.926703458911047e-02, -5.947410502199024e-02, 7.020775377840198e-03, 8.897687076368611e-03, 7.020775377840199e-03, 8.897687076368616e-03, 2.005276610245685e-04, 2.081708555177485e-04, 1.144646111319692e-05, 8.228450632683886e-06, 7.126652922888744e-05, 8.062626476070766e-05, 7.503434100857373e-03, 9.430102977001873e-03, 9.726881190240534e-05, 2.209771784165726e-04, 9.726881190240530e-05, 2.209771784165720e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05