
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_bmk_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_bmk", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.010101183971166e-01, -1.010123241459620e-01, -1.010186911400951e-01, -1.009859511027199e-01, -1.010035925148934e-01, -1.010035925148934e-01, -5.489243574968998e-02, -5.489937685273578e-02, -5.506096427947242e-02, -5.478364531618656e-02, -5.490109370261355e-02, -5.490109370261355e-02, -2.601791403728681e-02, -2.589727692259182e-02, -2.367512128305737e-02, -2.380989252041079e-02, -2.383470817726662e-02, -2.383470817726662e-02, -1.029177708544606e-02, -1.086163864502870e-02, -2.867470075825634e-02, 1.092451072135801e-03, -5.086674210333172e-03, -5.086674210333406e-03, 5.954483774511741e-03, 6.228022556453158e-03, 2.199909589995984e-02, 3.629492117911914e-03, 4.469071984287392e-03, 4.469071984287385e-03, -9.498750892623500e-02, -9.554826173260697e-02, -9.501058206178048e-02, -9.550556128398081e-02, -9.527241373134167e-02, -9.527241373134167e-02, -3.313289465988310e-02, -3.333818511766501e-02, -3.279797438232196e-02, -3.296434608640933e-02, -3.342794988704709e-02, -3.342794988704709e-02, -3.509416752897752e-02, -8.350306426021036e-02, -3.133351969767324e-02, -6.830302001018199e-02, -3.809854456620174e-02, -3.809854456620174e-02, 1.782635080633615e-02, -4.434445336583003e-03, 1.936166774309717e-02, -1.224087150724241e-01, 8.156649516053341e-03, 8.156649516053341e-03, 2.854749487416014e-03, 3.554162514482896e-03, 2.767788918430747e-03, 2.328234793210514e-02, 3.291689096703248e-03, 3.291689096703261e-03, -8.470288071235861e-02, -8.253790908889821e-02, -8.903499840355911e-02, -9.180424623622595e-02, -9.092476262022113e-02, -9.092476262022113e-02, -5.331900197049103e-02, -2.348844780548293e-02, -2.930962682047729e-02, -4.443337209341611e-02, -3.499366147092664e-02, -3.499366147092664e-02, -8.157740648554306e-02, -1.076024811614555e-02, -1.476814706776007e-02, -1.946098370659312e-02, -1.679618042669716e-02, -1.679618042669681e-02, -2.253870563502127e-02, 2.169340949530028e-02, 2.421572625277157e-02, -2.174089941563596e-02, 1.887520732428546e-02, 1.887520732428544e-02, 8.034040510372004e-03, 1.020448856131286e-03, 2.072289436596374e-03, 2.020411364193065e-02, 3.063073834600325e-03, 3.063073834600331e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_bmk_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_bmk", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.007103257439611e-01, -3.007182857663625e-01, -3.007411972300124e-01, -3.006230369530739e-01, -3.006867196062291e-01, -3.006867196062291e-01, -1.497721819891970e-01, -1.498033317860689e-01, -1.505268517324223e-01, -1.492791888090269e-01, -1.498104370074256e-01, -1.498104370074256e-01, -5.330559415686823e-02, -5.260354087868673e-02, -3.889000616509646e-02, -3.946581340731260e-02, -3.972459437692358e-02, -3.972459437692358e-02, -3.241113594319922e-02, -3.028763133998401e-02, -6.343731694949120e-02, -4.996579530881525e-02, -4.419910697798176e-02, -4.419910697798064e-02, 7.644979081135731e-03, 7.977678795592035e-03, 1.920029493107036e-02, 4.729445422812639e-03, 5.791633385202016e-03, 5.791633385202002e-03, -2.866897000547292e-01, -2.881789620657630e-01, -2.867512914524190e-01, -2.880661076177446e-01, -2.874482643896289e-01, -2.874482643896289e-01, -5.159413084023682e-02, -5.267667340291944e-02, -4.952381976991706e-02, -5.034942673035642e-02, -5.335456201947419e-02, -5.335456201947419e-02, -1.013435065506669e-01, -1.839049078586224e-01, -8.661803805218414e-02, -1.815317407358286e-01, -1.130689397294837e-01, -1.130689397294837e-01, -2.911009122526656e-02, -5.015871587426824e-02, -2.337527198950335e-02, -2.552437939990739e-01, -4.522304145994825e-02, -4.522304145994825e-02, 3.734418815457845e-03, 4.631953341115636e-03, 3.611814280158158e-03, 2.192008925640765e-03, 4.287248456139831e-03, 4.287248456139929e-03, -1.843974211198888e-02, -1.752393755165384e-01, -1.487661487399083e-01, -1.057898514635981e-01, -1.301664427894894e-01, -1.301664427894894e-01, 6.487374813278311e-02, -4.928279754184656e-02, -7.846759619258466e-02, -1.356259636572510e-01, -1.025926640960650e-01, -1.025926640960650e-01, -1.915712796339356e-01, -3.940490106017066e-02, -2.679449794300383e-02, -3.671723774292450e-02, -2.441345551809570e-02, -2.441345551809315e-02, -4.645883173921541e-02, 2.001179659599014e-02, 1.486665235406343e-02, -5.146321716812629e-02, -1.857196814233106e-02, -1.857196814233077e-02, 1.016550475577577e-02, 1.348040180934200e-03, 2.720990712815213e-03, -1.321526926692866e-02, 3.993617845321499e-03, 3.993617845321346e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_bmk_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_bmk", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.899899164726467e-10, 4.900118570233362e-10, 4.900616231556154e-10, 4.897362494122121e-10, 4.899135030945351e-10, 4.899135030945351e-10, 1.444219728061607e-06, 1.444877013967725e-06, 1.459996497075210e-06, 1.432534513181724e-06, 1.444842708184076e-06, 1.444842708184076e-06, 8.640337295393947e-04, 8.380720671721256e-04, 3.723858177230312e-04, 3.748120227539356e-04, 3.880258797980284e-04, 3.880258797980284e-04, 1.515075045722730e-01, 1.300815439012210e-01, 5.847486223010617e-04, 6.952532985936324e-01, 4.152717756042894e-01, 4.152717756042670e-01, 1.198282219109725e+01, 1.258601244972328e+01, 4.852908077279981e+00, 1.146136740310225e+01, 1.428334350290777e+01, 1.428334350291794e+01, 7.451319067231610e-07, 7.540570600919807e-07, 7.454929283532990e-07, 7.533711404032802e-07, 7.496691323205023e-07, 7.496691323205023e-07, 1.778435460100528e-06, 1.854334597825051e-06, 1.539982638271010e-06, 1.594288488932972e-06, 1.957292933379410e-06, 1.957292933379410e-06, 8.429843919020482e-03, 3.007463125674613e-02, 9.030805630647419e-03, 4.677839822245745e-02, 9.884205102711810e-03, 9.884205102711810e-03, 1.697554143085301e+00, 2.042625064684011e-01, 2.030229503636323e+00, 1.725995346044699e-04, 1.424697755096902e+00, 1.424697755096902e+00, 1.538513768451897e+01, 1.351200889627707e+01, 8.642754075407545e+01, 4.159758843532277e+00, 3.971484175490829e+01, 3.971484175453212e+01, -1.108486780245679e-01, 4.389161577466014e-02, 3.318647723223855e-02, 2.309018221564871e-03, 2.159255377793648e-02, 2.159255377793648e-02, -4.372490677740424e-01, 3.936642887817966e-03, 9.554347049130916e-03, 2.615745332923786e-02, 1.556648043285881e-02, 1.556648043285881e-02, 2.400851903672927e-02, 5.690846231140301e-02, 1.298952542219228e-02, 1.225007361765795e-02, 5.427858819453980e-03, 5.427858819460223e-03, 5.334224878800856e-03, 4.516125889949668e+00, 3.913129246264212e+00, 3.784310835151144e-02, 3.612676383453898e+00, 3.612676383453782e+00, 9.257678560793456e+00, 7.105178774467795e+01, 3.330686317630706e+01, 3.911921303647451e+00, 5.043931170788304e+01, 5.043931170822846e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05