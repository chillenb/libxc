
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_otpss_d_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.193348591968202e+01, -2.193353182500090e+01, -2.193380930468985e+01, -2.193312647603673e+01, -2.193347379977006e+01, -2.193347379977006e+01, -3.470959466726756e+00, -3.470971778687997e+00, -3.471558612912038e+00, -3.473449199737146e+00, -3.472123219595699e+00, -3.472123219595699e+00, -6.918759952084447e-01, -6.914529094696824e-01, -6.834221145534443e-01, -6.904571120296571e-01, -6.882815521695920e-01, -6.882815521695920e-01, -2.130881663419880e-01, -2.145363899214983e-01, -7.887263215654187e-01, -1.822689570702341e-01, -1.926365000299937e-01, -1.926365000299938e-01, -9.945104210018905e-03, -1.046718552926767e-02, -5.728135244485569e-02, -5.744817163152347e-03, -7.218250792989913e-03, -7.218250792989913e-03, -5.364630086003770e+00, -5.364972556847436e+00, -5.364686211482975e+00, -5.364987266397210e+00, -5.364782567237614e+00, -5.364782567237614e+00, -2.124626059910314e+00, -2.137202250642617e+00, -2.119078986379522e+00, -2.130167366597906e+00, -2.134250087141268e+00, -2.134250087141268e+00, -6.339040068876942e-01, -6.738397281626736e-01, -5.752098616417411e-01, -5.860802577869870e-01, -6.426360792542520e-01, -6.426360792542520e-01, -1.410226366902017e-01, -2.291644918653308e-01, -1.316390237688779e-01, -1.895704584267902e+00, -1.570074829839413e-01, -1.570074829839413e-01, -4.432646241990434e-03, -5.616046778262308e-03, -4.294096392583546e-03, -9.125614106816075e-02, -5.173062436562827e-03, -5.173062436562827e-03, -6.301578448374974e-01, -6.331388955420902e-01, -6.327675566631344e-01, -6.320194435546378e-01, -6.324525996606351e-01, -6.324525996606351e-01, -6.073771507273690e-01, -5.450971015539380e-01, -5.669959154822479e-01, -5.874301482429207e-01, -5.770904237739819e-01, -5.770904237739819e-01, -7.046449648242910e-01, -2.711183458087137e-01, -3.089792870878946e-01, -3.843506324366505e-01, -3.446734742820128e-01, -3.446734742820128e-01, -5.020186630629836e-01, -5.488700123531862e-02, -7.430872243768057e-02, -3.733412529592197e-01, -1.133025260742750e-01, -1.133025260742751e-01, -1.403217248637839e-02, -1.501333598160243e-03, -3.151261077550054e-03, -1.072017959723077e-01, -4.786727545338200e-03, -4.786727545338201e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_otpss_d_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.696044459878819e+01, -2.696051666515927e+01, -2.696093116656013e+01, -2.695986070910733e+01, -2.696040815997911e+01, -2.696040815997911e+01, -4.256662175944515e+00, -4.256803179170983e+00, -4.260703790934101e+00, -4.259889296325382e+00, -4.259369325209023e+00, -4.259369325209023e+00, -8.553803964007399e-01, -8.536679836745147e-01, -8.090011320782606e-01, -8.166956349703607e-01, -8.159558601219913e-01, -8.159558601219913e-01, -2.161943292954096e-01, -2.216584072298826e-01, -1.011583396625889e+00, -1.627612016618332e-01, -1.659996805677595e-01, -1.659996805677596e-01, -1.325195798841188e-02, -1.394643945009393e-02, -7.504474014122360e-02, -7.658255842631712e-03, -9.620977603056425e-03, -9.620977603056424e-03, -6.664972866252334e+00, -6.666018041864085e+00, -6.665075338690634e+00, -6.665995836787864e+00, -6.665474346308116e+00, -6.665474346308116e+00, -2.546631201846454e+00, -2.565706293512183e+00, -2.528908393589750e+00, -2.545813200643006e+00, -2.565148906685732e+00, -2.565148906685732e+00, -8.021604109341817e-01, -8.740097190967846e-01, -7.442154422703737e-01, -8.109396226691792e-01, -8.143140855735458e-01, -8.143140855735458e-01, -1.605166413294635e-01, -1.882240876044503e-01, -1.533608466732378e-01, -2.678369086829133e+00, -1.567075283066394e-01, -1.567075283066394e-01, -5.909404019308969e-03, -7.486541617084288e-03, -5.723757951572928e-03, -1.146926390929051e-01, -6.895384171585367e-03, -6.895384171585368e-03, -8.138744808145791e-01, -8.071049043923519e-01, -8.092818622956834e-01, -8.112740070916461e-01, -8.102560037690147e-01, -8.102560037690147e-01, -7.932354126343869e-01, -6.903022037510474e-01, -7.187463224754816e-01, -7.445751849702489e-01, -7.310940285534375e-01, -7.310940285534375e-01, -9.249895167616594e-01, -2.630587171751060e-01, -3.410666390879267e-01, -4.721040135196296e-01, -4.115133745762622e-01, -4.115133745762621e-01, -6.350698501458971e-01, -7.208035334962684e-02, -9.610641968634676e-02, -4.738807290902547e-01, -1.337428327061262e-01, -1.337428327061262e-01, -1.868887426497461e-02, -2.001713811705360e-03, -4.201267583075013e-03, -1.289945054692618e-01, -6.380493876229782e-03, -6.380493876229773e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.172689589980221e-09, -9.172891194246140e-09, -9.173331118908222e-09, -9.170345034945710e-09, -9.171974257549181e-09, -9.171974257549181e-09, -1.219366220875571e-05, -1.219889270770849e-05, -1.234650114492821e-05, -1.235647286416796e-05, -1.232893796307975e-05, -1.232893796307975e-05, -3.697231462836768e-03, -3.730829598336610e-03, -4.517688358996956e-03, -4.516860385346857e-03, -4.494812340539292e-03, -4.494812340539292e-03, -4.523330209574302e-01, -4.471097052961523e-01, -9.324195736344498e-04, -1.115425568969267e+00, -1.039068735427553e+00, -1.039068735427552e+00, -1.209306974311596e+00, -1.278255720763112e+00, -7.998145023733481e-01, -1.096257816773170e+00, -1.393918983445714e+00, -1.393918983448700e+00, -3.636166195535316e-06, -3.666338444521379e-06, -3.637612947673584e-06, -3.664239984234759e-06, -3.651376332721440e-06, -3.651376332721440e-06, -2.990820928529364e-05, -3.060932138580717e-05, -3.041982709170425e-05, -3.105132630100793e-05, -3.015851602248295e-05, -3.015851602248295e-05, -7.130320902392999e-03, -3.327163107699435e-03, -2.529930986019691e-02, -1.235636837121821e-02, -7.051701011480458e-03, -7.051701011480458e-03, -9.333872768719538e-01, -6.554424568239487e-01, -9.721702050124353e-01, -1.577469268376611e-04, -1.386375160038066e+00, -1.386375160038066e+00, -1.443974336080142e+00, -1.290245072204065e+00, -8.105638398497318e+00, -1.115080313698988e+00, -3.772239594523453e+00, -3.772239594531043e+00, -5.394241564755146e-02, -3.336927081004759e-02, -3.864436982734633e-02, -4.437467293655451e-02, -4.131420569118963e-02, -4.131420569118965e-02, -3.362783048071133e-02, -8.844584922718193e-03, -1.242191135891640e-02, -1.964461606304300e-02, -1.542759206633288e-02, -1.542759206633288e-02, -3.579293435842978e-03, -1.871964262156930e-01, -1.011708997708599e-01, -1.186946727546441e-01, -8.819327412354111e-02, -8.819327412354178e-02, -9.221341674646766e-03, -7.187664201641782e-01, -7.841435278710760e-01, -1.233151148753209e-01, -1.525062219935824e+00, -1.525062219935825e+00, -9.776478212705462e-01, -6.326399084350075e+00, -3.062669143793048e+00, -1.459112593725838e+00, -4.764299647141647e+00, -4.764299647190993e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.942141753534970e-04, 7.942541343330785e-04, 7.944227455673765e-04, 7.938317894683653e-04, 7.941426146238166e-04, 7.941426146238166e-04, 3.024733479128006e-03, 3.027810676166948e-03, 3.116668543144905e-03, 3.136273568779518e-03, 3.110711912720633e-03, 3.110711912720633e-03, 4.171698404052599e-03, 4.197133883429365e-03, 4.670929371400213e-03, 5.157854161082646e-03, 4.991577339993368e-03, 4.991577339993368e-03, -4.786671854733667e-03, -1.762786166400299e-03, 7.287217617172879e-04, 1.510824699069218e-03, 4.236635268271661e-03, 4.236635268271663e-03, 2.839250542834688e-09, 3.837315811922192e-09, 2.887872130929419e-06, 1.337234829268828e-10, 6.486844902985769e-10, 6.486844902977791e-10, 6.173124863409998e-03, 6.245027224536732e-03, 6.177759196568967e-03, 6.241176650356942e-03, 6.208750439194159e-03, 6.208750439194159e-03, -4.181825818166142e-04, -1.928021994520819e-04, -3.950350886170937e-04, -1.901032120695337e-04, -3.003845241917194e-04, -3.003845241917194e-04, 8.383962021732690e-03, -9.871459952307878e-04, 4.621422540446590e-02, 3.288307503367423e-02, 8.400865356173130e-03, 8.400865356173130e-03, 1.538566870946692e-04, 1.556403101200250e-02, 1.147211594152543e-04, 1.345809904726702e-02, 7.363534484899810e-04, 7.363534484899818e-04, 2.233990930491685e-11, 9.978149467307981e-11, 6.371442348045130e-10, 2.820183887482697e-05, 3.688808646064828e-10, 3.688808646054731e-10, 9.962237828941797e-02, 7.994963447698514e-02, 8.858592026177760e-02, 9.559328542758198e-02, 9.217207644654493e-02, 9.217207644654488e-02, 2.378891445297423e-02, 1.120030320938674e-02, 2.074831596008558e-02, 3.674455449315436e-02, 2.740814876357786e-02, 2.740814876357787e-02, 7.131292867835924e-03, -4.305393999157624e-03, 4.689312497636326e-03, 6.407977246006424e-02, 2.457832744051485e-02, 2.457832744051509e-02, 4.288260043949450e-03, 2.432915903428409e-06, 7.769661201794633e-06, 6.120167339395630e-02, 1.262275105524063e-04, 1.262275105524109e-04, 4.659305398956451e-09, 2.317525646413627e-13, 4.115973566790823e-11, 1.051928675829200e-04, 4.332964416300041e-10, 4.332964564579493e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05