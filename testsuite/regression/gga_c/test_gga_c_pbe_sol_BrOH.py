
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_sol_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_sol", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.169422397567999e-02, -6.169497598383703e-02, -6.169724596781558e-02, -6.168495414794767e-02, -6.169462962686619e-02, -6.169462962686619e-02, -5.081715788541569e-02, -5.082132276257618e-02, -5.093600616744037e-02, -5.073899331425569e-02, -5.081930852265251e-02, -5.081930852265251e-02, -3.286839776658678e-02, -3.264832340008435e-02, -2.681247138377561e-02, -2.712889390495648e-02, -3.278908345835365e-02, -3.278908345835365e-02, -1.294825487679810e-02, -1.375786044784627e-02, -3.999419512398349e-02, -4.131343666004920e-03, -1.319897987544516e-02, -1.319897987544516e-02, -2.606127279273762e-07, -3.221125857626939e-07, -6.474778138431415e-05, -6.429664597528585e-09, -3.192118250182654e-07, -3.192118250182654e-07, -6.822014320089635e-02, -6.841731875195285e-02, -6.823981237231423e-02, -6.839312674581818e-02, -6.832178451660646e-02, -6.832178451660646e-02, -2.831007319406499e-02, -2.881954980602293e-02, -2.760385693815679e-02, -2.799819427525099e-02, -2.959551137593278e-02, -2.959551137593278e-02, -4.454986912804402e-02, -5.891735356472819e-02, -4.312392903232955e-02, -5.432526721897458e-02, -4.680393838378879e-02, -4.680393838378879e-02, -1.174628756758346e-03, -6.980486966438715e-03, -1.286848529218634e-03, -7.720475311974907e-02, -2.458351482030576e-03, -2.458351482030576e-03, -5.775863009657103e-09, -1.072431618453082e-08, -8.079566571794598e-09, -2.690674171949924e-04, -1.031069366796745e-08, -1.031069366796745e-08, -6.234020427638413e-02, -5.859073403838237e-02, -5.988838537069009e-02, -6.092654226392665e-02, -6.040360259459602e-02, -6.040360259459602e-02, -6.224981096559286e-02, -3.518521517336732e-02, -4.186820190208455e-02, -4.894883142194613e-02, -4.530985374808368e-02, -4.530985374808369e-02, -5.920292627080348e-02, -1.118978744220148e-02, -1.703816000155452e-02, -3.002207894893489e-02, -2.323955802161726e-02, -2.323955802161725e-02, -3.338939438113455e-02, -3.909912720079198e-05, -1.451007077671388e-04, -3.422188804565286e-02, -8.284433759155720e-04, -8.284433759155789e-04, -3.997793241444154e-07, -4.219163878857375e-11, -7.574677607945932e-10, -8.380848073683103e-04, -7.085610611267935e-09, -7.085610608699012e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_sol_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_sol", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.393492393429843e-01, -1.393500460635463e-01, -1.393524945741771e-01, -1.393393077052500e-01, -1.393496743960354e-01, -1.393496743960354e-01, -1.189800143899041e-01, -1.189838159770778e-01, -1.190886848734033e-01, -1.189102027099570e-01, -1.189820154895825e-01, -1.189820154895825e-01, -8.366524362122170e-02, -8.347744780188963e-02, -7.734737828734781e-02, -7.781046080757913e-02, -8.359783799873914e-02, -8.359783799873914e-02, -4.493117088373922e-02, -4.646160691620907e-02, -9.118256375075204e-02, -1.971907406766535e-02, -4.541511455780561e-02, -4.541511455780561e-02, -1.689643910851108e-06, -2.086944661111382e-06, -4.020255047337490e-04, -4.202995748035529e-08, -2.068349310136263e-06, -2.068349310136263e-06, -1.367307559344300e-01, -1.368658719617021e-01, -1.367442783518256e-01, -1.368493444884629e-01, -1.368004799557055e-01, -1.368004799557055e-01, -8.782623680797907e-02, -8.868554690118390e-02, -8.664455482035915e-02, -8.732698448837796e-02, -8.991915151989012e-02, -8.991915151989012e-02, -8.461261663642018e-02, -8.075223007001091e-02, -8.323884769422343e-02, -8.030278739375736e-02, -8.541008021810759e-02, -8.541008021810759e-02, -6.520024741462598e-03, -3.027224754798133e-02, -7.084697832984771e-03, -1.167008764580815e-01, -1.266362734199428e-02, -1.266362734199428e-02, -3.776692767964505e-08, -7.004257542076145e-08, -5.288075850617423e-08, -1.612118315996347e-03, -6.738700825144805e-08, -6.738700824971457e-08, -7.468315643813400e-02, -7.839925708632214e-02, -7.721911641679111e-02, -7.619479140906565e-02, -7.671971676715740e-02, -7.671971676715740e-02, -7.303999694471740e-02, -7.994426138069764e-02, -8.199213321212112e-02, -8.145793687922193e-02, -8.205613812069845e-02, -8.205613812069845e-02, -8.267130790975893e-02, -4.272150991872760e-02, -5.531326544318551e-02, -7.060487864226951e-02, -6.426788108229109e-02, -6.426788108229108e-02, -7.714537359944841e-02, -2.446704864980883e-04, -8.856331606072995e-04, -6.988501327585729e-02, -4.701910339147827e-03, -4.701910339147868e-03, -2.586931278007254e-06, -2.779869689640579e-10, -4.971886089135251e-09, -4.748307670555507e-03, -4.636725534088913e-08, -4.636725534113874e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_sol_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_sol", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.962248159229878e-10, 1.962276214419289e-10, 1.962309201833817e-10, 1.961851722087631e-10, 1.962263724959693e-10, 1.962263724959693e-10, 1.080186252658701e-06, 1.080343603115411e-06, 1.084463108840985e-06, 1.076269250809548e-06, 1.080244814090465e-06, 1.080244814090465e-06, 1.685331957355114e-03, 1.676221189788563e-03, 1.416239855472862e-03, 1.387326606605604e-03, 1.682088841007329e-03, 1.682088841007329e-03, 2.434453010096156e-01, 2.483694418972971e-01, 9.553720732317641e-04, 2.322429245592487e-01, 2.452326432269115e-01, 2.452326432269115e-01, 6.183830851508928e-02, 6.661154926775824e-02, 1.324091154040399e-01, 1.160884602340841e-02, 6.942859163330949e-02, 6.942859163330949e-02, 2.727924744585847e-07, 2.743818629641395e-07, 2.729492816139572e-07, 2.741850468817644e-07, 2.736125168144958e-07, 2.736125168144958e-07, 6.860884246539439e-06, 6.814646603302923e-06, 6.690069220019525e-06, 6.656269558508323e-06, 7.055315829806939e-06, 7.055315829806939e-06, 5.377740887989102e-03, 5.979248875578273e-03, 6.560503481447799e-03, 8.004313469011856e-03, 4.875859257584085e-03, 4.875859257584085e-03, 1.696538636448573e-01, 9.752711266686233e-02, 2.072527738502030e-01, 5.004036593191673e-05, 2.445433043291612e-01, 2.445433043291612e-01, 1.196403230840974e-02, 1.489966125144892e-02, 4.510249461241625e-02, 2.060298619789877e-01, 2.301513526509552e-02, 2.301513526566439e-02, 8.551065147769965e-03, 7.759058175200820e-03, 8.023967073142423e-03, 8.242926353569302e-03, 8.131881296138080e-03, 8.131881296138080e-03, 9.960801784089739e-03, 7.360919634778596e-03, 7.853568075380563e-03, 8.471111136830802e-03, 8.157129906885288e-03, 8.157129906885290e-03, 4.810439288408676e-03, 6.205598542910060e-02, 4.858500150076125e-02, 3.310015211574929e-02, 4.149508728515578e-02, 4.149508728515579e-02, 1.119431739359543e-02, 9.201773202887137e-02, 1.400454763023806e-01, 5.271736063803638e-02, 2.881722261686528e-01, 2.881722261686553e-01, 5.109953358340832e-02, 3.744521412444575e-03, 9.100698921506467e-03, 3.716807116827694e-01, 3.113558754660054e-02, 3.113558754671451e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05