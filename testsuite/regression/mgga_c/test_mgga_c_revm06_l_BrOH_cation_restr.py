
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm06_l_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.190247896560821e-01, -1.190258465787977e-01, -1.190317153136816e-01, -1.190161199900663e-01, -1.190241225636312e-01, -1.190241225636312e-01, -3.502289105375897e-02, -3.503446904564749e-02, -3.533038705440325e-02, -3.511797850499394e-02, -3.517873667015822e-02, -3.517873667015822e-02, -6.546609889534462e-03, -6.344227544443026e-03, -2.129696332298814e-03, -2.742705012220584e-03, -2.653343825935940e-03, -2.653343825935940e-03, -7.136427615040762e-03, -7.791171027885724e-03, -2.501287234659988e-02, 2.058906673015988e-02, 8.293038712185230e-03, 8.293038712185234e-03, 9.096626816741851e-03, 9.558555022472779e-03, 3.675870563492308e-02, 5.915532698466955e-03, 7.089728528362479e-03, 7.089728528362485e-03, -9.789525833455431e-02, -9.803443800122114e-02, -9.790578232175015e-02, -9.802842096782320e-02, -9.796366939419718e-02, -9.796366939419718e-02, -4.051573022474375e-02, -4.285265349106747e-02, -4.146753563478142e-02, -4.368643631891132e-02, -4.136917023713588e-02, -4.136917023713588e-02, -3.236448155110219e-02, -5.275834853319931e-02, -2.288422024524118e-02, -4.125397377604203e-02, -3.505903159492767e-02, -3.505903159492767e-02, 4.077275715768437e-02, 6.201916267395472e-03, 4.241112347002433e-02, -6.878972699192290e-02, 2.631503326784715e-02, 2.631503326784714e-02, 5.396638993716652e-03, 6.231287061707016e-03, 4.562054594101654e-03, 4.359550489726971e-02, 5.696934173266862e-03, 5.696934173266858e-03, -6.286603657205549e-02, -5.715221337075622e-02, -5.891674067999195e-02, -6.055140544579860e-02, -5.971127812001094e-02, -5.971127812001094e-02, -6.415673180707895e-02, -3.540449768329656e-02, -3.910161412522339e-02, -4.541354418357612e-02, -4.161326965387024e-02, -4.161326965387024e-02, -5.093667445082174e-02, -6.938355830894194e-03, -1.288146200168614e-02, -1.419126860752775e-02, -1.534310437591326e-02, -1.534310437591327e-02, -2.380146319388985e-02, 3.496798231144796e-02, 4.205008924987384e-02, -1.841037269301367e-02, 3.916825037571015e-02, 3.916825037571016e-02, 1.405875813122232e-02, 2.256168126665745e-03, 3.278606893906456e-03, 3.874750206020504e-02, 5.145370859594653e-03, 5.145370859594663e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm06_l_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.332280986105362e-01, -1.332308935439438e-01, -1.332311046630850e-01, -1.331893358137145e-01, -1.332129668681774e-01, -1.332129668681774e-01, -1.321148545782427e-01, -1.321301618444109e-01, -1.324652613546657e-01, -1.316824095157620e-01, -1.320370870896683e-01, -1.320370870896683e-01, -5.078284111358813e-02, -5.011150961614785e-02, -3.504384419720454e-02, -3.426580941447323e-02, -3.509684162482639e-02, -3.509684162482639e-02, -1.165377375416377e-02, -9.580434345292772e-03, -5.653667819120389e-02, -2.808283506808412e-02, -2.652722064226963e-02, -2.652722064226967e-02, 1.755828960284869e-02, 1.833132153997438e-02, 5.514263211508882e-02, 1.105177941216826e-02, 1.343210737245509e-02, 1.343210737245509e-02, -1.137891025030554e-01, -1.142553053450371e-01, -1.137911727617050e-01, -1.142035592716548e-01, -1.140356644522973e-01, -1.140356644522973e-01, 2.776445115558850e-03, 3.613731442338404e-03, 7.650117217397237e-03, 8.696839043859622e-03, 6.457531677746056e-04, 6.457531677746056e-04, -7.397938379837284e-02, -9.408339225940521e-02, -7.350265899638260e-02, -9.177695323370097e-02, -7.861475112173891e-02, -7.861475112173891e-02, 1.219306467580012e-02, -2.733052732743646e-02, 1.902226167434933e-02, -1.288025159088787e-01, -1.426831910417590e-02, -1.426831910417583e-02, 8.985557431675277e-03, 1.095413210666859e-02, 8.523450118313883e-03, 4.493210910447391e-02, 1.014997133983572e-02, 1.014997133983571e-02, -8.093141820336501e-02, -8.194547744876461e-02, -8.240706480636070e-02, -8.213027361926772e-02, -8.234899839636881e-02, -8.234899839636881e-02, -7.916758239129670e-02, -1.159774703374215e-02, -3.816638335713946e-02, -6.804841961317569e-02, -5.389019578501062e-02, -5.389019578501062e-02, -9.680964157044859e-02, -1.589718247967618e-02, -1.114558539367757e-03, -2.275646884027608e-02, 7.825481740660126e-04, 7.825481740660403e-04, -2.465028670534806e-02, 5.496092486532241e-02, 5.584175506192335e-02, -4.002624970096406e-02, 2.189437857781902e-02, 2.189437857781920e-02, 2.376422580687497e-02, 3.414429821011520e-03, 6.409456732798557e-03, 2.805099728744351e-02, 9.426541662339289e-03, 9.426541662339303e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_l_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.938274438148934e-10, -2.938168741992116e-10, -2.938139677294921e-10, -2.939722190574104e-10, -2.938830046325876e-10, -2.938830046325876e-10, 6.442138260348343e-07, 6.443026414149237e-07, 6.442888809137777e-07, 6.218975812046749e-07, 6.338950860162772e-07, 6.338950860162772e-07, 1.560810503151614e-03, 1.527872312016115e-03, 7.404873484047523e-04, 4.363843346384349e-04, 5.633957059138533e-04, 5.633957059138533e-04, -6.152082324411080e-01, -6.186541646195616e-01, 2.229351789462799e-03, -1.022298783356091e-01, -3.437278728341622e-01, -3.437278728341620e-01, -2.291249504352106e+03, -2.088403316158067e+03, -2.512572401361352e+01, -6.504935686775242e+03, -4.674083173897574e+03, -4.674083173897572e+03, -2.109835122746867e-07, -2.080461917663406e-07, -2.109654683361654e-07, -2.083683986748314e-07, -2.094384369431779e-07, -2.094384369431779e-07, -2.451194733523650e-05, -2.491894294985108e-05, -2.535961448968154e-05, -2.578428499306051e-05, -2.430122161070097e-05, -2.430122161070097e-05, -7.263365513371894e-04, 6.506480078720286e-03, 2.083828470129911e-03, 1.205192348534710e-02, -1.998089235274330e-05, -1.998089235274330e-05, -6.261748352218084e-01, -2.389044413945644e-01, -8.903896039709399e-01, 3.810565432812493e-05, -4.843547126133704e-01, -4.843547126133718e-01, -9.454220283508892e+03, -6.420060977225362e+03, -3.249027613186702e+04, -5.345824324046813e+00, -1.348879776001767e+04, -1.348879776001766e+04, 1.643111109725713e-02, 6.887135712005714e-03, 9.824728576585241e-03, 1.258688229297440e-02, 1.116612611245188e-02, 1.116612611245187e-02, 1.408059274156686e-02, -2.104225525669310e-02, -1.524168708449771e-02, -5.033817655642354e-03, -1.062851447163414e-02, -1.062851447163414e-02, 5.299796762194639e-03, -1.930704787475842e-01, -1.410681885079962e-01, -4.169942951871307e-02, -1.033305243259018e-01, -1.033305243259019e-01, -1.909931178091609e-02, -2.912854054588571e+01, -1.099571133402978e+01, -3.932441084549652e-02, -2.175882227826178e+00, -2.175882227826199e+00, -7.341217234014979e+02, -1.169155159489560e+05, -4.150061863492005e+04, -3.414322582701595e+00, -1.903484777162581e+04, -1.903484777162593e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_l_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_l_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.213570110353847e-05, 7.213499939996986e-05, 7.213670776684198e-05, 7.214728902927133e-05, 7.214105569011650e-05, 7.214105569011650e-05, 4.055452149966435e-04, 4.056870159899867e-04, 4.098735572189521e-04, 4.125331079684202e-04, 4.104629521431692e-04, 4.104629521431692e-04, -1.504837299676419e-05, 2.855427537118848e-06, 7.038185815135808e-04, 1.493445886507192e-03, 1.196845489072939e-03, 1.196845489072939e-03, 7.889644410406922e-02, 7.906613633320896e-02, -3.698891725012093e-03, 3.720425366943491e-02, 5.042284599906843e-02, 5.042284599906842e-02, 7.952812947520172e-03, 8.401542388486090e-03, 1.886775025161418e-02, 3.893453155972002e-03, 5.818425972103180e-03, 5.818425972103173e-03, 8.701621358313626e-04, 8.656438868314203e-04, 8.701372163298236e-04, 8.661417889332060e-04, 8.677876021166091e-04, 8.677876021166091e-04, 3.169323386503795e-03, 3.313481029311383e-03, 3.235507939422638e-03, 3.373456882438175e-03, 3.215448687221862e-03, 3.215448687221862e-03, 1.863147566561486e-02, 1.083424585562646e-02, 1.273639605055883e-02, 8.759070977919409e-03, 1.838795511169646e-02, 1.838795511169646e-02, 3.003234735110116e-02, 4.861214305435902e-02, 2.892054698905571e-02, 1.281194775419439e-03, 4.583820282986647e-02, 4.583820282986649e-02, 1.883440531176486e-03, 3.112340018552386e-03, 7.995127716400220e-03, 2.594185214702147e-02, 5.271382308413016e-03, 5.271382308413018e-03, -4.122451528854433e-03, 1.043688530504987e-02, 4.901019658441540e-03, 2.548168643589588e-04, 2.554303384538616e-03, 2.554303384538616e-03, 1.220038783433501e-02, 4.650153215775134e-02, 4.444915694266247e-02, 3.392517793618158e-02, 4.021497858333795e-02, 4.021497858333794e-02, 9.256640643429161e-03, 5.319526466599394e-02, 5.049795054781626e-02, 3.370124343186802e-02, 4.834376983474707e-02, 4.834376983474709e-02, 3.385943308675879e-02, 1.955561324891722e-02, 2.082410559533601e-02, 4.084326386577957e-02, 3.644971481301382e-02, 3.644971481301389e-02, 5.556937305063658e-03, 5.398896832675584e-04, 4.314996054453954e-03, 3.901886293472488e-02, 6.258428294695409e-03, 6.258428294695424e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05