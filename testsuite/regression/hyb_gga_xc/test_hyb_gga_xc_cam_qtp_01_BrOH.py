
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_qtp_01_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_01", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.615343597670765e+01, -1.615345430159245e+01, -1.615359549802206e+01, -1.615329423536405e+01, -1.615344514372899e+01, -1.615344514372899e+01, -2.609342179150861e+00, -2.609322313406110e+00, -2.608887620967166e+00, -2.610208692058826e+00, -2.609343451556768e+00, -2.609343451556768e+00, -4.369328811626135e-01, -4.366580161591824e-01, -4.299026327008031e-01, -4.337296345454266e-01, -4.368313443446397e-01, -4.368313443446397e-01, -6.149387179844362e-02, -6.347816036055731e-02, -5.341198960962787e-01, -2.467830813692692e-02, -6.209796212778028e-02, -6.209796212778028e-02, -2.574532839369821e-03, -2.684744503759423e-03, -2.141706906290862e-03, -1.312793679806839e-03, -2.659453183943080e-03, -2.659453183943080e-03, -3.825907959673229e+00, -3.825468572342213e+00, -3.825867891525777e+00, -3.825526164664004e+00, -3.825675729884328e+00, -3.825675729884328e+00, -1.527291541465038e+00, -1.535538453480296e+00, -1.527292512850454e+00, -1.533718963475706e+00, -1.532874634077526e+00, -1.532874634077526e+00, -3.560742787059651e-01, -3.830609726920621e-01, -3.340871087701588e-01, -3.440183819041284e-01, -3.715372705322622e-01, -3.715372705322622e-01, -3.393609639340388e-03, -5.562797241796154e-02, -4.689434564967866e-03, -1.329231306881413e+00, -1.342283887349828e-02, -1.342283887349828e-02, -1.269640661428066e-03, -1.437129632465296e-03, -1.098916936355279e-03, 9.148257301416742e-05, -1.318317555733493e-03, -1.318317555733493e-03, -3.567812196808404e-01, -3.570504872866236e-01, -3.569836809320636e-01, -3.569087689457899e-01, -3.569483586808451e-01, -3.569483586808451e-01, -3.424415374528719e-01, -3.023295618761024e-01, -3.151518484795868e-01, -3.261737757971290e-01, -3.206018123028155e-01, -3.206018123028155e-01, -4.059770545888107e-01, -9.218595519171370e-02, -1.271938576779517e-01, -1.852140972992547e-01, -1.551138542143379e-01, -1.551138542143378e-01, -2.643391070168709e-01, -2.305536025036544e-03, 8.769268133222137e-04, -1.694985131798454e-01, -2.537807857928872e-03, -2.537807857928889e-03, -2.970813940891609e-03, -4.226772685696622e-04, -7.793152003401085e-04, -3.503775078642828e-03, -1.132941029281513e-03, -1.132941029281512e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_qtp_01_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_01", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.938809757281038e+01, -1.938816070266904e+01, -1.938845392581418e+01, -1.938741992972601e+01, -1.938813076849002e+01, -1.938813076849002e+01, -3.094327417402649e+00, -3.094349616272592e+00, -3.095089056391222e+00, -3.094479242514023e+00, -3.094352176968113e+00, -3.094352176968113e+00, -5.529709471855380e-01, -5.524060050821838e-01, -5.389974922658936e-01, -5.433665329149202e-01, -5.527643366340287e-01, -5.527643366340287e-01, -1.196040651384010e-01, -1.209049534173723e-01, -6.725034940928213e-01, -8.580755562489291e-02, -1.199769320662369e-01, -1.199769320662369e-01, -3.351125265881050e-03, -3.493292978682760e-03, -1.376631124657589e-02, -1.719367260141875e-03, -3.460676325430162e-03, -3.460676325430162e-03, -4.737251591599345e+00, -4.739032174980100e+00, -4.737434191695447e+00, -4.738818673700210e+00, -4.738163619494625e+00, -4.738163619494625e+00, -1.684620598168027e+00, -1.697339743912526e+00, -1.678810058706024e+00, -1.688695306113701e+00, -1.700891730706142e+00, -1.700891730706142e+00, -4.739118541484126e-01, -5.252261158712753e-01, -4.464341703020139e-01, -4.708400856978538e-01, -4.948589111840286e-01, -4.948589111840286e-01, -6.138971417840464e-02, -1.344638958699634e-01, -5.857753512597059e-02, -1.738437958602785e+00, -7.031662787141624e-02, -7.031662787141624e-02, -1.663338533820395e-03, -1.880685998892249e-03, -1.441451903926947e-03, -2.697238355536716e-02, -1.726537752215908e-03, -1.726537752215908e-03, -4.977319523099875e-01, -4.925937502983381e-01, -4.943549021237191e-01, -4.957727061051683e-01, -4.950566036067442e-01, -4.950566036067442e-01, -4.796872086875351e-01, -4.033413059583576e-01, -4.229019429462303e-01, -4.428143691334814e-01, -4.324425555803000e-01, -4.324425555803001e-01, -5.538987990305658e-01, -1.735260593948631e-01, -2.056420899575939e-01, -2.647552123378970e-01, -2.321676508301203e-01, -2.321676508301202e-01, -3.585747364432549e-01, -1.181386349629994e-02, -2.242505027856766e-02, -2.442390745611848e-01, -4.182169701570863e-02, -4.182169701570858e-02, -3.861794666031346e-03, -5.580205265129485e-04, -1.024957979977667e-03, -3.754447822754429e-02, -1.485702338267920e-03, -1.485702338267918e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_qtp_01_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_01", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.501987484285892e-09, -5.501964570521265e-09, -5.501779383855467e-09, -5.502156273659036e-09, -5.501976093841890e-09, -5.501976093841890e-09, -6.627529795698757e-06, -6.627754773788281e-06, -6.632881501020822e-06, -6.618600310194137e-06, -6.627535818483484e-06, -6.627535818483484e-06, -2.177770039085041e-03, -2.173218856433008e-03, -2.025217253758459e-03, -1.996285346729311e-03, -2.176166624558581e-03, -2.176166624558581e-03, 1.660415472410872e-01, 1.534763402192200e-01, -1.398497352464240e-03, 6.955323766740564e-01, 1.621636707924508e-01, 1.621636707924508e-01, 2.581754080744533e-03, 5.199162154427130e-03, 8.071577341614256e+00, -2.335477116663404e-05, 4.456540529164652e-03, 4.456540529164652e-03, -1.576801200034312e-06, -1.577952031681436e-06, -1.576909885153850e-06, -1.577804814324406e-06, -1.577403184845482e-06, -1.577403184845482e-06, -4.506995960557820e-05, -4.428525393702732e-05, -4.499129005671209e-05, -4.438034728800402e-05, -4.463868326550178e-05, -4.463868326550178e-05, -4.419365038363158e-03, -4.586633548317585e-03, -5.040661767695555e-03, -5.726327350347328e-03, -4.113988333302867e-03, -4.113988333302867e-03, 1.943581996230275e+00, 1.794248160991532e-01, 2.016236912372309e+00, -8.597308799941471e-05, 1.232288328262519e+00, 1.232288328262519e+00, -2.204294111581920e-05, -3.175367465730914e-05, -2.963404325562168e-05, 5.818960461490295e+00, -3.214372647469699e-05, -3.214372647454390e-05, -6.062637200166730e-03, -5.585678848700332e-03, -5.732971176349941e-03, -5.862449959448793e-03, -5.795744937638321e-03, -5.795744937638321e-03, -6.862351155932958e-03, -5.317022728185899e-03, -5.644955296395473e-03, -5.966000120392206e-03, -5.807759752588855e-03, -5.807759752588856e-03, -3.888456908850945e-03, 5.186284873583198e-02, 1.134043901728268e-02, -8.629209381330914e-03, -2.592357770550042e-03, -2.592357770550009e-03, -6.544428217274319e-03, 8.138375095188159e+00, 6.768717202590063e+00, -1.182177560654154e-02, 3.394672888591241e+00, 3.394672888591240e+00, 2.291600177331504e-02, -1.284397226813848e-06, -6.919788264267721e-06, 3.789538585127511e+00, -2.671624267389638e-05, -2.671624267344769e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05