
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_mohlyp2_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.120806153579555e+01, -2.120809733428636e+01, -2.120831746256608e+01, -2.120773007202803e+01, -2.120807990959896e+01, -2.120807990959896e+01, -3.518845447747200e+00, -3.518825215012770e+00, -3.518414820429114e+00, -3.519864666265618e+00, -3.518849921437344e+00, -3.518849921437344e+00, -7.188528472252063e-01, -7.190191101327090e-01, -7.288516831521392e-01, -7.327172668806874e-01, -7.189081641926384e-01, -7.189081641926384e-01, -2.417201963085720e-01, -2.408392337574520e-01, -8.346846449684781e-01, -2.299492223608746e-01, -2.413952569708216e-01, -2.413952569708216e-01, -2.943602522813560e-02, -3.081478642762457e-02, -1.129571234995846e-01, -1.419120490258204e-02, -3.049436425652334e-02, -3.049436425652334e-02, -5.154494749982629e+00, -5.154763212173162e+00, -5.154526421579456e+00, -5.154735054143132e+00, -5.154625526930937e+00, -5.154625526930937e+00, -2.168713880196975e+00, -2.176023392741613e+00, -2.173832921613858e+00, -2.179361517298883e+00, -2.167531670406881e+00, -2.167531670406881e+00, -5.973934280904576e-01, -6.373802931087851e-01, -5.687353292133145e-01, -5.828073875505646e-01, -6.174086498723452e-01, -6.174086498723452e-01, -2.031170979349802e-01, -2.858227761158857e-01, -1.989993917793929e-01, -1.902433435709059e+00, -2.138431127205848e-01, -2.138431127205848e-01, -1.369091722771239e-02, -1.564052930819478e-02, -1.172510028410029e-02, -1.451653786273878e-01, -1.425150062993215e-02, -1.425150062993215e-02, -6.056214990826028e-01, -6.028017630693776e-01, -6.037399558955618e-01, -6.045185498937786e-01, -6.041227205231674e-01, -6.041227205231674e-01, -5.866001690360223e-01, -5.340749498060070e-01, -5.441308110388758e-01, -5.573184887927081e-01, -5.500246064219001e-01, -5.500246064219001e-01, -6.677916021037350e-01, -3.205605969524475e-01, -3.409352601393558e-01, -3.810668579490883e-01, -3.562864123679788e-01, -3.562864123679787e-01, -4.853979954267759e-01, -1.063835731158502e-01, -1.348444524403158e-01, -3.507964583545422e-01, -1.727629531528924e-01, -1.727629531528924e-01, -3.445667448957206e-02, -4.288980380014840e-03, -8.143862856633752e-03, -1.653864351745841e-01, -1.211488165763272e-02, -1.211488165763270e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_mohlyp2_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.665638752325214e+01, -2.665650313555741e+01, -2.665699274701254e+01, -2.665509999689396e+01, -2.665644871200735e+01, -2.665644871200735e+01, -4.236054147019648e+00, -4.236146615720408e+00, -4.238856917451425e+00, -4.235072476290894e+00, -4.236119607460858e+00, -4.236119607460858e+00, -7.516603389826074e-01, -7.487579652344775e-01, -6.673620892526871e-01, -6.759394967625454e-01, -7.506129506653927e-01, -7.506129506653927e-01, -1.551272901864649e-01, -1.597343674222305e-01, -9.603692185737325e-01, -1.421363468078063e-01, -1.564864909356166e-01, -1.564864909356166e-01, -3.881330059736113e-02, -4.058699032874529e-02, -1.377296285457278e-01, -1.887949113678988e-02, -4.016599923249375e-02, -4.016599923249375e-02, -6.641349994994779e+00, -6.644761694820663e+00, -6.641698199412845e+00, -6.644351152602531e+00, -6.643103440143365e+00, -6.643103440143365e+00, -1.957724572089137e+00, -1.986669400301636e+00, -1.930628310646343e+00, -1.953229344620477e+00, -2.012542202510226e+00, -2.012542202510226e+00, -7.407912113518442e-01, -8.416696857243889e-01, -6.995366195940754e-01, -7.643385217540959e-01, -7.769760285246191e-01, -7.769760285246191e-01, -1.827325613592338e-01, -1.615381254797266e-01, -1.733966887745746e-01, -2.513946958585021e+00, -1.543580112925235e-01, -1.543580112925235e-01, -1.821559404392021e-02, -2.079628867213138e-02, -1.559275771088262e-02, -1.620293400796480e-01, -1.894916565654536e-02, -1.894916565654534e-02, -8.031576392047117e-01, -7.969347026303851e-01, -7.994359563099177e-01, -8.011640725127110e-01, -8.003222134861069e-01, -8.003222134861069e-01, -7.780095243396274e-01, -5.989282847042335e-01, -6.640288445052749e-01, -7.166938916979715e-01, -6.913683145776235e-01, -6.913683145776235e-01, -8.809457560300763e-01, -1.903584966847767e-01, -2.480275911544683e-01, -4.056927568615961e-01, -3.202707289431355e-01, -3.202707289431354e-01, -5.341240842004408e-01, -1.326475923874322e-01, -1.601912623385267e-01, -4.098566384450504e-01, -1.628477179254373e-01, -1.628477179254371e-01, -4.533632258533628e-02, -5.716935077942836e-03, -1.084827657035549e-02, -1.534357824145451e-01, -1.611401248376196e-02, -1.611401248376194e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_mohlyp2_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp2", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.078006954760497e-09, -4.077890911500343e-09, -4.077454683767734e-09, -4.079353381925447e-09, -4.077945076486398e-09, -4.077945076486398e-09, -7.157033994662945e-06, -7.156221308448370e-06, -7.132699923365776e-06, -7.167156942328492e-06, -7.156494522112156e-06, -7.156494522112156e-06, -6.919061543889555e-03, -6.980456585008092e-03, -8.576576600219264e-03, -8.283274995552444e-03, -6.941301113479973e-03, -6.941301113479973e-03, -1.292196334822620e+00, -1.246954267612765e+00, -2.862631635979031e-03, -2.432601661700892e+00, -1.279122967675488e+00, -1.279122967675488e+00, -1.894788748686441e+01, -1.899451345466654e+01, -2.737595493120227e+00, -1.343811613402297e+01, -1.973279137573291e+01, -1.973279137573291e+01, -8.983088416170357e-07, -8.920568599500558e-07, -8.976804144443721e-07, -8.928189771737645e-07, -8.950871275557255e-07, -8.950871275557255e-07, -1.071559973487698e-04, -1.036370663325059e-04, -1.091530261976782e-04, -1.063742011503969e-04, -1.021550330300218e-04, -1.021550330300218e-04, -7.589988597103414e-03, -1.432014155833176e-03, -9.797518354436777e-03, -3.389762563196815e-03, -5.879118935635837e-03, -5.879118935635837e-03, -2.633102236377163e+00, -9.216387453422945e-01, -3.116822790411812e+00, -2.115544162638486e-05, -3.066616855397862e+00, -3.066616855397862e+00, -1.425598473350411e+01, -1.428417235350130e+01, -4.088488617937956e+01, -3.769142689421568e+00, -2.109836757453307e+01, -2.109836757453806e+01, -5.250818419221133e-05, -1.485803165139770e-03, -9.798167989734624e-04, -5.825560119480933e-04, -7.818189573444456e-04, -7.818189573444458e-04, 2.415122575017727e-04, -1.900972469708682e-02, -1.230911087569858e-02, -6.816004865675523e-03, -9.472243491056359e-03, -9.472243491056352e-03, -1.339772812679860e-03, -4.703588622680206e-01, -2.660904211338077e-01, -8.631732434820197e-02, -1.602005895964686e-01, -1.602005895964688e-01, -2.963858430195323e-02, -1.531539194315434e+00, -2.252141292472247e+00, -8.996885048284506e-02, -4.620101495693053e+00, -4.620101495693069e+00, -1.421788936657153e+01, -2.457710745020951e+01, -2.115981401263919e+01, -5.901227474492789e+00, -3.081389406028239e+01, -3.081389406027924e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05