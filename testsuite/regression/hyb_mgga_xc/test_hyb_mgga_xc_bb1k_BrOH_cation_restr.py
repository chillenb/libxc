
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_bb1k_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.229185714039270e+01, -1.229187291089085e+01, -1.229198240822098e+01, -1.229174693171248e+01, -1.229186471638737e+01, -1.229186471638737e+01, -2.088045294451712e+00, -2.088022915893939e+00, -2.087536851798059e+00, -2.088566297855516e+00, -2.087971248218099e+00, -2.087971248218099e+00, -4.449790526050489e-01, -4.447782529504418e-01, -4.409499827309804e-01, -4.432732426025698e-01, -4.425375877235179e-01, -4.425375877235179e-01, -1.368189934289158e-01, -1.376441408092320e-01, -5.271437407553381e-01, -1.180121007448351e-01, -1.244651744550715e-01, -1.244651744550715e-01, -3.765279724870648e-02, -3.764564356342318e-02, -6.619425912311569e-02, -3.457516773648257e-02, -3.433046196939999e-02, -3.433046196939997e-02, -2.992883294199087e+00, -2.992513392117553e+00, -2.992871381885366e+00, -2.992544792156215e+00, -2.992692979816519e+00, -2.992692979816519e+00, -1.269178709044598e+00, -1.275076215447806e+00, -1.269226309023354e+00, -1.274424764589337e+00, -1.272421523438551e+00, -1.272421523438551e+00, -3.713430680709165e-01, -3.904961554619580e-01, -3.494302548467472e-01, -3.563089933221863e-01, -3.751978396899440e-01, -3.751978396899440e-01, -9.976662056216183e-02, -1.469949216615224e-01, -9.534875136708668e-02, -1.121513024805184e+00, -1.036455209486636e-01, -1.036455209486636e-01, -3.123407857073337e-02, -3.330126133213769e-02, -2.177804416353259e-02, -7.710296219263925e-02, -2.634310094548958e-02, -2.634310094548959e-02, -3.535487096207752e-01, -3.549774181087845e-01, -3.545006199527844e-01, -3.540837052199755e-01, -3.542946598077240e-01, -3.542946598077240e-01, -3.446460073842892e-01, -3.229655055937122e-01, -3.293353319450921e-01, -3.353691724342972e-01, -3.322575733815618e-01, -3.322575733815618e-01, -4.104190865746720e-01, -1.735532538525970e-01, -1.959482280230161e-01, -2.364586391176979e-01, -2.140711207814541e-01, -2.140711207814541e-01, -3.002557916747367e-01, -6.625546673708083e-02, -7.366961745122812e-02, -2.241402212394525e-01, -8.337115812412255e-02, -8.337115812412252e-02, -4.231048343996525e-02, -1.934419367080209e-02, -2.523759219188490e-02, -8.117337739709267e-02, -2.476048298856613e-02, -2.476048298856611e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_bb1k_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.473759391848047e+01, -1.473764857992595e+01, -1.473788119989304e+01, -1.473706812169086e+01, -1.473749563331408e+01, -1.473749563331408e+01, -2.457722363749977e+00, -2.457738039397202e+00, -2.458123067766349e+00, -2.457513046155813e+00, -2.457612620044971e+00, -2.457612620044971e+00, -5.236480170374708e-01, -5.227573442197933e-01, -5.000806375874816e-01, -5.020292930601098e-01, -5.022427231342925e-01, -5.022427231342925e-01, -1.358090166096685e-01, -1.383595501725072e-01, -6.437463303207701e-01, -1.012394645802219e-01, -1.149932701044588e-01, -1.149932701044587e-01, -1.042495010410889e-02, -1.062974424428165e-02, -3.143651260213180e-02, -8.126618195639734e-03, -8.742566776084128e-03, -8.742566776084154e-03, -3.682021835799734e+00, -3.683422973194778e+00, -3.682083947504762e+00, -3.683320941970049e+00, -3.682733142415898e+00, -3.682733142415898e+00, -1.351652345566091e+00, -1.361870719730630e+00, -1.341450097542017e+00, -1.350433824414443e+00, -1.362277505716821e+00, -1.362277505716821e+00, -4.521713704259384e-01, -4.939526316276873e-01, -4.245055669640626e-01, -4.449872634900115e-01, -4.595484202389986e-01, -4.595484202389986e-01, -6.944606955858036e-02, -1.319565198319939e-01, -6.450682555902580e-02, -1.417654303302190e+00, -8.175895087545418e-02, -8.175895087545418e-02, -6.981582628647202e-03, -7.865534529198824e-03, -5.422705666175977e-03, -4.509508402570366e-02, -6.550145483980472e-03, -6.550145483980497e-03, -4.586421819529790e-01, -4.545973125496005e-01, -4.560495196478808e-01, -4.572250023360730e-01, -4.566397580198847e-01, -4.566397580198847e-01, -4.468677450761475e-01, -3.733775873459512e-01, -3.948112834345118e-01, -4.164181961207457e-01, -4.055149348404671e-01, -4.055149348404671e-01, -5.172169473601617e-01, -1.669696294583393e-01, -2.025830579064229e-01, -2.754674087024762e-01, -2.356477165153845e-01, -2.356477165153845e-01, -3.488577261232901e-01, -3.069663903474322e-02, -3.837261836404021e-02, -2.692419164752031e-01, -5.561218761686341e-02, -5.561218761686340e-02, -1.280996427404994e-02, -3.653826439454865e-03, -5.426372295880130e-03, -5.243288063702976e-02, -6.123425308279306e-03, -6.123425308279271e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_bb1k_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.065816105413517e-09, -4.065795227589040e-09, -4.065652713014156e-09, -4.065964530474274e-09, -4.065808201074577e-09, -4.065808201074577e-09, -4.386076868346309e-06, -4.386238062663353e-06, -4.390313803140025e-06, -4.388216425020119e-06, -4.389738064578970e-06, -4.389738064578970e-06, -1.233409503284951e-03, -1.257123780635451e-03, -1.797250412778770e-03, -1.812666114488335e-03, -1.790499981786769e-03, -1.790499981786769e-03, -3.126206209904185e-01, -2.978502425720955e-01, 1.794131530321191e-04, -7.286004409623539e-01, -5.177904508450095e-01, -5.177904508450095e-01, -5.877899152680804e+03, -5.154275532745726e+03, -3.403515755858621e+01, -2.774255976394287e+04, -1.537427666528194e+04, -1.537427666528195e+04, -1.124178857438513e-06, -1.124645157725675e-06, -1.124199724993340e-06, -1.124611131018559e-06, -1.124415472315093e-06, -1.124415472315093e-06, -3.685240029934264e-05, -3.611886712940473e-05, -3.714614744989706e-05, -3.649101119686582e-05, -3.630393326272391e-05, -3.630393326272391e-05, -2.905614944948991e-03, -4.187108877045766e-04, -2.865633984641442e-03, 2.683608304204421e-03, -2.679819002426620e-03, -2.679819002426620e-03, -2.113821927418625e+00, -2.761344649253059e-01, -2.699935050218973e+00, -3.100752063296133e-05, -1.427465307107855e+00, -1.427465307107855e+00, -6.327897244410127e+04, -3.099556287382446e+04, -1.123173953870073e+05, -9.134026448881780e+00, -5.302739448653477e+04, -5.302739448653474e+04, -2.268095235455330e-03, -2.821264918047509e-03, -2.634320250879687e-03, -2.471378932249331e-03, -2.553349855089614e-03, -2.553349855089614e-03, -1.544862429203613e-03, -7.446720671207728e-03, -6.059136460484426e-03, -4.490946854965819e-03, -5.297716508983896e-03, -5.297716508983896e-03, -1.537703642467084e-04, -1.257791267418768e-01, -6.777001665651057e-02, -2.077486410255068e-02, -4.094043396555148e-02, -4.094043396555148e-02, -9.087623476688633e-03, -3.745214470471507e+01, -1.570921459825945e+01, -1.899696178841413e-02, -4.806643824903603e+00, -4.806643824903603e+00, -2.059804685672297e+03, -2.177516989834331e+06, -2.088509148633819e+05, -5.752445763128333e+00, -7.078973029750639e+04, -7.078973029750664e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_bb1k_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_bb1k_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.453792283978471e-05, -1.453818662584219e-05, -1.453879561912927e-05, -1.453488296950892e-05, -1.453701315010626e-05, -1.453701315010626e-05, -2.775718393848247e-04, -2.776240936498513e-04, -2.788234851368839e-04, -2.766509746532341e-04, -2.776228584892479e-04, -2.776228584892479e-04, -2.566454190939865e-03, -2.533371885065624e-03, -1.804185727185526e-03, -1.813146546056280e-03, -1.838213172349539e-03, -1.838213172349539e-03, -2.555797814259009e-03, -2.812007899408073e-03, -2.283128744728795e-03, -9.589339535762655e-04, -1.555106140527363e-03, -1.555106140527362e-03, -3.258796894617046e-07, -3.975288282833202e-07, -2.492603401189000e-05, -6.277512067909956e-08, -1.524265052365561e-07, -1.524265052365564e-07, -2.269702985532985e-04, -2.280573663706722e-04, -2.270144524712743e-04, -2.279741091301697e-04, -2.275237198577517e-04, -2.275237198577517e-04, -2.646821746696447e-04, -2.698698241827689e-04, -2.483762463982107e-04, -2.527232363479489e-04, -2.759354158000614e-04, -2.759354158000614e-04, -6.369241708602159e-03, -1.190292719782858e-02, -6.412473447449897e-03, -1.282699419954386e-02, -6.819996236865639e-03, -6.819996236865639e-03, -2.855271797554048e-04, -1.081443713068554e-03, -2.474314630730053e-04, -1.928645477872851e-03, -6.389134607717199e-04, -6.389134607717198e-04, -3.950316868999526e-08, -6.927478027200196e-08, -2.025449349540546e-07, -1.121481555037163e-04, -1.603352375432031e-07, -1.603352375432027e-07, -1.715653870777054e-02, -1.386575190372423e-02, -1.490949299207765e-02, -1.586232602812442e-02, -1.537527207624292e-02, -1.537527207624292e-02, -1.924636234980064e-02, -4.390069527485070e-03, -6.462968674229307e-03, -9.827025162610095e-03, -7.946372527416517e-03, -7.946372527416517e-03, -1.066059606991729e-02, -1.552655076970212e-03, -2.480934733989058e-03, -6.305605209934963e-03, -4.033158041228320e-03, -4.033158041228323e-03, -4.854046670396275e-03, -2.008518762128271e-05, -4.752137100784840e-05, -9.965204787316353e-03, -2.644635392811541e-04, -2.644635392811544e-04, -6.882113065106681e-07, -7.619803347053765e-09, -3.151297056615350e-08, -2.192720144245662e-04, -1.623227288495196e-07, -1.623227288495198e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05