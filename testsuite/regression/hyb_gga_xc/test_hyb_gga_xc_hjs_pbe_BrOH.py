
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hjs_pbe_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.577568100882623e+01, -1.577569963738700e+01, -1.577583902925960e+01, -1.577553286073520e+01, -1.577569036240090e+01, -1.577569036240090e+01, -2.661995701008381e+00, -2.661977852264463e+00, -2.661595142741436e+00, -2.662807670519243e+00, -2.661997622070392e+00, -2.661997622070392e+00, -5.650154835264430e-01, -5.648446676243075e-01, -5.620331748410299e-01, -5.655073486758591e-01, -5.649511555126807e-01, -5.649511555126807e-01, -1.867853781581695e-01, -1.880959621463139e-01, -6.577951287946142e-01, -1.525014030388256e-01, -1.871653394428507e-01, -1.871653394428507e-01, -1.696506216377979e-02, -1.776002229768469e-02, -6.517816487590278e-02, -8.171790577101167e-03, -1.757638046680573e-02, -1.757638046680573e-02, -3.848830248321014e+00, -3.848585174206424e+00, -3.848809397224744e+00, -3.848618753337362e+00, -3.848698253160250e+00, -3.848698253160250e+00, -1.621657669110103e+00, -1.629213945163537e+00, -1.622245404628934e+00, -1.628117163894263e+00, -1.626058028592828e+00, -1.626058028592828e+00, -4.847440660643747e-01, -5.180628393785486e-01, -4.629089378791559e-01, -4.768986578504081e-01, -5.004296051787138e-01, -5.004296051787138e-01, -1.238460997154001e-01, -1.977321630163833e-01, -1.219313507538785e-01, -1.445818848014199e+00, -1.359896802879272e-01, -1.359896802879272e-01, -7.883645833913506e-03, -9.007160625152546e-03, -6.752865164927728e-03, -8.524811180684741e-02, -8.207711187857066e-03, -8.207711187857066e-03, -4.954337521451089e-01, -4.925336305327818e-01, -4.934974140544052e-01, -4.942970395266050e-01, -4.938905000178112e-01, -4.938905000178112e-01, -4.814353877543836e-01, -4.318605958831012e-01, -4.440955920146467e-01, -4.567130128150436e-01, -4.500579620067836e-01, -4.500579620067836e-01, -5.405091585338488e-01, -2.343312133422558e-01, -2.640156186734435e-01, -3.147428813696936e-01, -2.873504406010163e-01, -2.873504406010163e-01, -3.944244327059930e-01, -6.123399903882142e-02, -7.863538643873798e-02, -2.967951076439821e-01, -1.042658180015740e-01, -1.042658180015740e-01, -1.984737914000844e-02, -2.468686465732011e-03, -4.688426450429481e-03, -9.995545905383763e-02, -6.976958075405901e-03, -6.976958075405887e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hjs_pbe_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.899916168789366e+01, -1.899922764905673e+01, -1.899952643940466e+01, -1.899844622089757e+01, -1.899919643575354e+01, -1.899919643575354e+01, -3.152389573193012e+00, -3.152423457447944e+00, -3.153480112809464e+00, -3.152303196003207e+00, -3.152419928923860e+00, -3.152419928923860e+00, -6.561354778893704e-01, -6.549720135225305e-01, -6.250741696491251e-01, -6.301250808212177e-01, -6.557136159421562e-01, -6.557136159421562e-01, -2.055306974595018e-01, -2.086367287014799e-01, -7.887504392511374e-01, -1.594507329280229e-01, -2.064586450038078e-01, -2.064586450038078e-01, -2.247084576631330e-02, -2.350667259794115e-02, -8.012248611279406e-02, -1.088242538389749e-02, -2.326517952947387e-02, -2.326517952947387e-02, -4.792572559190413e+00, -4.794758829811100e+00, -4.792795611522784e+00, -4.794495561458976e+00, -4.793694213511733e+00, -4.793694213511733e+00, -1.725370071698632e+00, -1.738847765959450e+00, -1.718152499376222e+00, -1.728564846069236e+00, -1.744239084089414e+00, -1.744239084089414e+00, -6.081627920204634e-01, -6.727151991717051e-01, -5.791691624187246e-01, -6.165550921216439e-01, -6.321231021016951e-01, -6.321231021016951e-01, -1.363862030214066e-01, -2.049127137780056e-01, -1.336990586156969e-01, -1.882020493973416e+00, -1.446375265663055e-01, -1.446375265663055e-01, -1.049932004379204e-02, -1.199122192762713e-02, -8.991904105165883e-03, -1.002914030809120e-01, -1.092742259645163e-02, -1.092742259645161e-02, -6.435123109160044e-01, -6.396107492804937e-01, -6.412250874575144e-01, -6.423177141534607e-01, -6.417893387273376e-01, -6.417893387273376e-01, -6.247614420990519e-01, -5.212991572364779e-01, -5.542565129300945e-01, -5.842335246565554e-01, -5.692671382855977e-01, -5.692671382855978e-01, -7.016913036092609e-01, -2.470583047287769e-01, -2.907332467920528e-01, -3.791031059554623e-01, -3.317307411965779e-01, -3.317307411965781e-01, -4.748133078114447e-01, -7.613798872235555e-02, -9.451274261673628e-02, -3.690986080549166e-01, -1.168668024298274e-01, -1.168668024298275e-01, -2.623794271615880e-02, -3.291115273373555e-03, -6.248235335533618e-03, -1.119847362369792e-01, -9.290980348372816e-03, -9.290980348372786e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hjs_pbe_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.096881768111328e-09, -5.096846863070166e-09, -5.096634613882676e-09, -5.097207312099313e-09, -5.096863832948797e-09, -5.096863832948797e-09, -6.054244150542277e-06, -6.054232603454887e-06, -6.052941240449668e-06, -6.050220037902531e-06, -6.054139202108799e-06, -6.054139202108799e-06, -2.861128926748596e-03, -2.876149929953731e-03, -3.206826879371557e-03, -3.119239217178217e-03, -2.866601772449644e-03, -2.866601772449644e-03, -2.817161476694051e-01, -2.696436677873409e-01, -1.415076223137280e-03, -5.802637548801505e-01, -2.781809298320750e-01, -2.781809298320750e-01, -5.033181198522067e+00, -5.039697991943968e+00, -1.703506657995823e+00, -3.603798835065015e+00, -5.236453518098677e+00, -5.236453518098677e+00, -1.276804274603826e-06, -1.275148079773550e-06, -1.276634931014250e-06, -1.275347361025808e-06, -1.275959043876570e-06, -1.275959043876570e-06, -4.905487592396407e-05, -4.804558317806324e-05, -4.913317139323404e-05, -4.835279230835460e-05, -4.823328755445773e-05, -4.823328755445773e-05, -3.271733793554501e-03, 1.018446138247083e-04, -4.024257286363166e-03, -7.475382115800840e-04, -2.616949396125798e-03, -2.616949396125798e-03, -7.764111641754775e-01, -2.220405100890146e-01, -8.801748501458345e-01, -2.948534518116744e-05, -7.697065041796551e-01, -7.697065041796551e-01, -3.823636005083349e+00, -3.827776072182827e+00, -1.095708923848890e+01, -1.541677831954726e+00, -5.653561042928112e+00, -5.653561042933045e+00, 2.004101075682132e-03, 5.135580188145045e-04, 9.992102692835386e-04, 1.410252908742521e-03, 1.200632492573919e-03, 1.200632492573916e-03, 2.639575926166387e-03, -7.079029899727807e-03, -4.843084609011115e-03, -2.491650139202816e-03, -3.715074605534691e-03, -3.715074605534687e-03, -1.191181820012150e-04, -1.158391811607250e-01, -6.815097421877368e-02, -2.468552573258517e-02, -4.307415024895819e-02, -4.307415024895829e-02, -1.023874401946643e-02, -1.469439828872606e+00, -1.371597273316925e+00, -2.265634086210906e-02, -1.359941345663221e+00, -1.359941345663207e+00, -3.763883290188049e+00, -6.619928087528570e+00, -5.687746554292476e+00, -1.665661601008219e+00, -8.260456585016847e+00, -8.260456585006738e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05