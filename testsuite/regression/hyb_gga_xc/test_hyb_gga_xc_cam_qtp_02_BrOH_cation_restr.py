
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_qtp_02_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_02", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.507906056543139e+01, -1.507908028894787e+01, -1.507921723111455e+01, -1.507892276864236e+01, -1.507907005665689e+01, -1.507907005665689e+01, -2.422654411327395e+00, -2.422630647169675e+00, -2.422142960200516e+00, -2.423469020872028e+00, -2.422686614527056e+00, -2.422686614527056e+00, -3.970222800059716e-01, -3.966968462879713e-01, -3.890222693899908e-01, -3.929026889379363e-01, -3.918546131244233e-01, -3.918546131244233e-01, -3.965435153203289e-02, -4.192507273994832e-02, -4.776804821305872e-01, -9.751236788667893e-03, -2.195206790883986e-02, -2.195206790883983e-02, -1.047062816187083e-03, -1.101099324614194e-03, 5.307184767382173e-05, -6.093337431084453e-04, -7.635585387220254e-04, -7.635585387220254e-04, -3.560867893938292e+00, -3.560453182152018e+00, -3.560856458449417e+00, -3.560490242682059e+00, -3.560653528524835e+00, -3.560653528524835e+00, -1.410548088487870e+00, -1.418600787841191e+00, -1.409408246544485e+00, -1.416537181926143e+00, -1.415534586657251e+00, -1.415534586657251e+00, -3.267883540570136e-01, -3.534916890804784e-01, -2.967101895327650e-01, -3.048041679405612e-01, -3.324806401925532e-01, -3.324806401925532e-01, 1.291638543731460e-02, -3.001600194974989e-02, 1.347499780697134e-02, -1.231362365022800e+00, 1.208100813172639e-03, 1.208100813172639e-03, -4.713423117794051e-04, -5.958245795681151e-04, -4.567719481234037e-04, 8.634598707795507e-03, -5.493210774638267e-04, -5.493210774638267e-04, -3.184082951968887e-01, -3.183746675520675e-01, -3.184167446469019e-01, -3.184248698592666e-01, -3.184232016937580e-01, -3.184232016937580e-01, -3.069680537866811e-01, -2.643823494600129e-01, -2.778420410138355e-01, -2.900956154520287e-01, -2.839202189137809e-01, -2.839202189137809e-01, -3.750081582829417e-01, -6.283060897992683e-02, -9.767244559440316e-02, -1.605299277587281e-01, -1.279074487646286e-01, -1.279074487646286e-01, -2.360281523187565e-01, -3.733473548463484e-04, 6.172779402189642e-03, -1.516270447749242e-01, 8.281557093863173e-03, 8.281557093863069e-03, -1.468016979156744e-03, -1.605944058920394e-04, -3.359432540760249e-04, 8.482530624809012e-03, -5.086767296619888e-04, -5.086767296619884e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_qtp_02_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_02", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.810625670123944e+01, -1.810632444148550e+01, -1.810661376957109e+01, -1.810560618256914e+01, -1.810613581530797e+01, -1.810613581530797e+01, -2.880613413046850e+00, -2.880637249066009e+00, -2.881260449494779e+00, -2.880732848569794e+00, -2.880712479608003e+00, -2.880712479608003e+00, -5.149618395940021e-01, -5.144261855379260e-01, -5.027463624607681e-01, -5.070711434046210e-01, -5.059929990107059e-01, -5.059929990107059e-01, -1.076222779041582e-01, -1.085416480355839e-01, -6.081159439333836e-01, -8.444237233559279e-02, -9.211899322517103e-02, -9.211899322517103e-02, -1.390144246601584e-03, -1.461655588857428e-03, -6.370140818611245e-03, -8.102132903756465e-04, -1.014693273817876e-03, -1.014693273817876e-03, -4.415245588836798e+00, -4.416949658024601e+00, -4.415322899676920e+00, -4.416827247921266e+00, -4.416109698888478e+00, -4.416109698888478e+00, -1.571032110693169e+00, -1.583118068382304e+00, -1.561486714827241e+00, -1.572085458416192e+00, -1.582413494585772e+00, -1.582413494585772e+00, -4.412984195124081e-01, -4.877643343158733e-01, -4.038033994660362e-01, -4.227583800264614e-01, -4.494241626760217e-01, -4.494241626760217e-01, -5.932141658419052e-02, -1.234008285207231e-01, -5.240920741353066e-02, -1.614520283153307e+00, -6.570668660412675e-02, -6.570668660412675e-02, -6.270810966471997e-04, -7.922926242870356e-04, -6.077337394159640e-04, -2.311815284065029e-02, -7.305906147075120e-04, -7.305906147075120e-04, -4.471499202155117e-01, -4.426540743248266e-01, -4.441572737207342e-01, -4.454577012626376e-01, -4.447996593224152e-01, -4.447996593224152e-01, -4.334531725463644e-01, -3.635409365709156e-01, -3.805941125064628e-01, -3.996068228825810e-01, -3.896000307760200e-01, -3.896000307760200e-01, -5.146986399754638e-01, -1.574339300925029e-01, -1.853432676524484e-01, -2.407236423539149e-01, -2.092769912907902e-01, -2.092769912907901e-01, -3.303768612797411e-01, -5.730634198286530e-03, -1.334547480559875e-02, -2.262511139131557e-01, -3.633782166074563e-02, -3.633782166074562e-02, -1.946968147575488e-03, -2.139561074980207e-04, -4.472061557616218e-04, -3.254045033077189e-02, -6.766463176275756e-04, -6.766463176275751e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_qtp_02_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_02", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.133517991990964e-09, -5.133493490164831e-09, -5.133314847274011e-09, -5.133680836384522e-09, -5.133498969335835e-09, -5.133498969335835e-09, -6.102978350232217e-06, -6.103249431078078e-06, -6.108945049543765e-06, -6.094574512238113e-06, -6.102734353673147e-06, -6.102734353673147e-06, -1.757248508015791e-03, -1.752053777850687e-03, -1.599394157959170e-03, -1.578658663775898e-03, -1.593396738374340e-03, -1.593396738374340e-03, 3.035731940183165e-01, 2.830747190887741e-01, -1.180074584814141e-03, 9.117861846575300e-01, 5.890506581671270e-01, 5.890506581671274e-01, -2.158706973278920e-05, -2.558364699125881e-05, 9.577842978279643e+00, -4.495046210940381e-06, -9.947168523450753e-06, -9.947168523353177e-06, -1.459007544093337e-06, -1.460118817073989e-06, -1.459044409419979e-06, -1.460025559211681e-06, -1.459577906099714e-06, -1.459577906099714e-06, -4.060133751792606e-05, -3.989335757250104e-05, -4.054086878948529e-05, -3.991598005945021e-05, -4.023879324444589e-05, -4.023879324444589e-05, -3.356496656497853e-03, -3.640221043323948e-03, -3.884533635505410e-03, -4.904933153859100e-03, -3.343051811421743e-03, -3.343051811421743e-03, 2.915319112457154e+00, 3.136487313496233e-01, 3.588950328961920e+00, -7.794549655731060e-05, 1.814268138201695e+00, 1.814268138201695e+00, -2.650675394357772e-06, -4.725339762081084e-06, -8.003007628620733e-06, 7.992685254852780e+00, -7.893302179644517e-06, -7.893302179841298e-06, -5.176215238714736e-03, -4.697385919752220e-03, -4.843539381740857e-03, -4.980372053760796e-03, -4.909891317389238e-03, -4.909891317389238e-03, -5.861877836051617e-03, -3.632933016528672e-03, -4.224381791731456e-03, -4.768060234467295e-03, -4.502283937025092e-03, -4.502283937025092e-03, -3.106217344885334e-03, 1.077231502048985e-01, 3.746960165660428e-02, -1.164973025867603e-03, 1.131136688613502e-02, 1.131136688613503e-02, -3.971146608976230e-03, 9.189291845316447e+00, 9.880476199015012e+00, -2.677949826841594e-03, 5.183007768963411e+00, 5.183007768963400e+00, 1.257301586880109e-04, -3.588898342977500e-07, -1.729715422455754e-06, 5.904848299373702e+00, -7.486433605911586e-06, -7.486433605527108e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05