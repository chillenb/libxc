
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_sb98_1b_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1b", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.602880042253701e+01, -1.602883554868473e+01, -1.602902569481766e+01, -1.602844983372745e+01, -1.602881866733544e+01, -1.602881866733544e+01, -2.702558292657465e+00, -2.702555579042078e+00, -2.702589522009437e+00, -2.703087438422182e+00, -2.702568064588130e+00, -2.702568064588130e+00, -5.736985411609771e-01, -5.734497473778635e-01, -5.687654390396482e-01, -5.723185326835972e-01, -5.736060820805745e-01, -5.736060820805745e-01, -1.893215617707976e-01, -1.904036834156997e-01, -6.703641488603198e-01, -1.597175416911976e-01, -1.896262925318092e-01, -1.896262925318092e-01, -1.638620464551603e-02, -1.716935366086717e-02, -6.909038513026772e-02, -7.838180178524314e-03, -1.698763140245766e-02, -1.698763140245766e-02, -3.975115014122292e+00, -3.975644294666091e+00, -3.975171681459249e+00, -3.975583173246818e+00, -3.975382542652095e+00, -3.975382542652095e+00, -1.610448980134737e+00, -1.618608955667556e+00, -1.609972677639536e+00, -1.616263421219982e+00, -1.616819362153717e+00, -1.616819362153717e+00, -4.936335906349419e-01, -5.265451818540148e-01, -4.710198825851643e-01, -4.832823614299262e-01, -5.097874781387801e-01, -5.097874781387801e-01, -1.347530401157521e-01, -2.054230225135675e-01, -1.318438825593942e-01, -1.506628809140777e+00, -1.444861612105921e-01, -1.444861612105921e-01, -7.561134960095606e-03, -8.642169019812601e-03, -6.475073533291041e-03, -9.214446890855722e-02, -7.872214144229566e-03, -7.872214144229572e-03, -5.089209470564080e-01, -4.998129603725001e-01, -5.017216816518895e-01, -5.039946817903888e-01, -5.027441567803388e-01, -5.027441567803388e-01, -4.968831861040204e-01, -4.389748269510443e-01, -4.515583526766093e-01, -4.634285810803723e-01, -4.573025775678803e-01, -4.573025775678803e-01, -5.501469461559170e-01, -2.402600306811203e-01, -2.683252768292985e-01, -3.186204035358253e-01, -2.911115181561524e-01, -2.911115181561524e-01, -4.005070221414749e-01, -6.462892284207016e-02, -8.502276333648755e-02, -2.991654090413210e-01, -1.122995562994180e-01, -1.122995562994180e-01, -1.924189153103148e-02, -2.379960102936186e-03, -4.500114431435055e-03, -1.069048734333165e-01, -6.690133633169822e-03, -6.690133633169799e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_sb98_1b_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1b", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.097083130815315e+01, -2.097089873556732e+01, -2.097121255850578e+01, -2.097010811571344e+01, -2.097086675832825e+01, -2.097086675832825e+01, -3.443144005455570e+00, -3.443177700769526e+00, -3.444240622888524e+00, -3.443120821772204e+00, -3.443175685003949e+00, -3.443175685003949e+00, -6.832041163226144e-01, -6.819311047103630e-01, -6.446888860347775e-01, -6.506059127914351e-01, -6.827436690135544e-01, -6.827436690135544e-01, -1.966758737231880e-01, -2.005918310575111e-01, -8.259404361728031e-01, -1.387503331415854e-01, -1.978744336292522e-01, -1.978744336292522e-01, -2.174642544402627e-02, -2.277055986536125e-02, -8.242715913988520e-02, -1.044079403698967e-02, -2.252899969365686e-02, -2.252899969365686e-02, -5.215127840646890e+00, -5.216771924831636e+00, -5.215297352649354e+00, -5.216575726593707e+00, -5.215969489825707e+00, -5.215969489825707e+00, -1.802758261904500e+00, -1.821863988282419e+00, -1.787822447039884e+00, -1.802859782752963e+00, -1.834673237900111e+00, -1.834673237900111e+00, -6.192237513514464e-01, -6.933809989210743e-01, -5.887206285909192e-01, -6.255652946006754e-01, -6.431908866737045e-01, -6.431908866737045e-01, -1.224135154115801e-01, -1.820191990795577e-01, -1.192582525033695e-01, -1.995267516597227e+00, -1.253452118381886e-01, -1.253452118381886e-01, -1.007110425778031e-02, -1.151145638188368e-02, -8.617205824763408e-03, -9.880636670103145e-02, -1.048211400981802e-02, -1.048211400981801e-02, -6.853537138967712e-01, -6.609722031860557e-01, -6.691554156986779e-01, -6.763139571588531e-01, -6.726722367541909e-01, -6.726722367541909e-01, -6.672404390631563e-01, -5.334028520415696e-01, -5.625329277795127e-01, -5.888899274059257e-01, -5.752509852162196e-01, -5.752509852162195e-01, -7.224346697499061e-01, -2.306737020538379e-01, -2.845975100655071e-01, -3.806718713237799e-01, -3.316532659494892e-01, -3.316532659494891e-01, -4.835159304514879e-01, -7.907491278525568e-02, -9.603708630354447e-02, -3.655402548414278e-01, -1.072648676739857e-01, -1.072648676739858e-01, -2.551029296619776e-02, -3.165921411636995e-03, -5.989803713573826e-03, -1.025090141245477e-01, -8.905445127501650e-03, -8.905445127501617e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_sb98_1b_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1b", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.065386446702599e-10, -9.064991052986677e-10, -9.063602913064981e-10, -9.070070499085919e-10, -9.065174787996363e-10, -9.065174787996363e-10, -2.133215039322738e-06, -2.132950670892764e-06, -2.125328790136056e-06, -2.136605066503449e-06, -2.133041707688654e-06, -2.133041707688654e-06, -2.294333046994628e-03, -2.311634881625369e-03, -2.838857161147014e-03, -2.738861319491220e-03, -2.300584250264827e-03, -2.300584250264827e-03, -3.912690216677436e-01, -3.703109016248363e-01, -1.012194301660719e-03, -1.087883296226341e+00, -3.848517086400900e-01, -3.848517086400900e-01, -8.381403738887618e+00, -8.470901087004762e+00, -4.223825344677389e+00, -5.253585168393644e+00, -8.781057670532279e+00, -8.781057670532279e+00, -2.268745617099231e-07, -2.246857575432790e-07, -2.266553409614244e-07, -2.249534682701205e-07, -2.257477349912880e-07, -2.257477349912880e-07, -3.722812015008766e-05, -3.581889928279391e-05, -3.821500118484800e-05, -3.708657716725286e-05, -3.502649305237013e-05, -3.502649305237013e-05, -3.792603335711479e-03, 1.448086461658750e-03, -4.718648479995884e-03, -2.626967862210162e-03, -3.259445460951856e-03, -3.259445460951856e-03, -1.842096456929537e+00, -3.721377836538155e-01, -2.045161965571165e+00, 9.412628425605731e-06, -1.608854654189019e+00, -1.608854654189019e+00, -5.541298858809516e+00, -5.671839944810455e+00, -1.549344969035912e+01, -3.882280777296106e+00, -8.247218775507065e+00, -8.247218775501841e+00, 3.445671892660154e-02, 3.863655817953873e-03, 9.407596933619495e-03, 1.674739610873802e-02, 1.262881026056063e-02, 1.262881026056062e-02, 5.601493860085842e-02, -6.807013507328953e-03, -5.737923515402084e-03, -5.064521508129647e-03, -5.454604002125708e-03, -5.454604002125697e-03, 4.626292044411089e-04, -1.668250123522016e-01, -8.453012085142984e-02, -2.917432656914522e-02, -4.938732981457992e-02, -4.938732981457997e-02, -1.038725114339918e-02, -3.625823506043484e+00, -3.517721639178065e+00, -3.788856746899893e-02, -3.231036328287101e+00, -3.231036328287103e+00, -6.488741565364208e+00, -8.226431665433584e+00, -7.627565109442642e+00, -3.891810391597560e+00, -1.173994433392594e+01, -1.173994433393582e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05