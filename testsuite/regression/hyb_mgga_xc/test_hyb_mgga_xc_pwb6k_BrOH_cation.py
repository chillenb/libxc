
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_pwb6k_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pwb6k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.159310211652334e+01, -1.159311431508560e+01, -1.159320997389226e+01, -1.159302759958440e+01, -1.159311727198389e+01, -1.159311727198389e+01, -1.975613705804232e+00, -1.975589599896760e+00, -1.975062114169072e+00, -1.976157561132656e+00, -1.975542126583599e+00, -1.975542126583599e+00, -4.226817249645980e-01, -4.225208227860947e-01, -4.194866598783190e-01, -4.217028844032326e-01, -4.221053503026445e-01, -4.221053503026445e-01, -1.293589576995228e-01, -1.303264948143075e-01, -4.998449531740038e-01, -1.075614567213105e-01, -1.202107966877434e-01, -1.202107966877433e-01, -6.724066787237821e-05, -8.161062752177014e-05, -1.227726496272144e-02, -1.131289954506549e-05, -3.691583491513330e-05, -3.691583491513356e-05, -2.818053162757866e+00, -2.817553889515037e+00, -2.818035709537865e+00, -2.817594901544497e+00, -2.817797101583821e+00, -2.817797101583821e+00, -1.208226485598989e+00, -1.213737485484773e+00, -1.208506818293589e+00, -1.213380049751938e+00, -1.211133305296478e+00, -1.211133305296478e+00, -3.517715841510810e-01, -3.677625414841392e-01, -3.313436464324386e-01, -3.362895738327937e-01, -3.554475825345387e-01, -3.554475825345388e-01, -7.781340308741529e-02, -1.366040218557966e-01, -7.139835603873466e-02, -1.053176449912350e+00, -9.049954465548261e-02, -9.049954465548261e-02, -5.841337639581819e-06, -1.149010604126919e-05, -1.334885014758960e-05, -4.078301245630994e-02, -1.879398293188092e-05, -1.879398293188089e-05, -3.319822355707001e-01, -3.341816902377584e-01, -3.334754435681002e-01, -3.328458740723931e-01, -3.331668307370065e-01, -3.331668307370065e-01, -3.233356626545569e-01, -3.071641651616602e-01, -3.124768064912171e-01, -3.171203887117462e-01, -3.147538928003192e-01, -3.147538928003192e-01, -3.866201933379370e-01, -1.637778684716113e-01, -1.863005668706813e-01, -2.250296178763085e-01, -2.039471640233584e-01, -2.039471640233584e-01, -2.855789705855036e-01, -1.032061243237527e-02, -2.382883856989087e-02, -2.130256573200129e-01, -6.058912441778141e-02, -6.058912441778142e-02, -1.769604015896275e-04, -4.473591604275789e-07, -3.020428816123593e-06, -5.573860121604129e-02, -1.560592544455180e-05, -1.560592544455171e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_pwb6k_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pwb6k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.365912232689696e+01, -1.365910106751388e+01, -1.365917998983568e+01, -1.365914333112643e+01, -1.365935534165590e+01, -1.365939500731204e+01, -1.365869938203615e+01, -1.365856454385475e+01, -1.365915386352081e+01, -1.365889135643290e+01, -1.365915386352081e+01, -1.365889135643290e+01, -2.285854400791414e+00, -2.285939261151020e+00, -2.285862990154726e+00, -2.285955112542598e+00, -2.286165281654845e+00, -2.286256639040911e+00, -2.285653591276330e+00, -2.285790934392362e+00, -2.285529359261474e+00, -2.286036814245622e+00, -2.285529359261474e+00, -2.286036814245622e+00, -4.908429479541658e-01, -4.923933630207674e-01, -4.898499093459593e-01, -4.918349433222466e-01, -4.730892950536262e-01, -4.701719256431024e-01, -4.731256461176063e-01, -4.735797045752369e-01, -4.880776778622058e-01, -4.630879439835835e-01, -4.880776778622058e-01, -4.630879439835835e-01, -1.368458610624757e-01, -1.350308557817828e-01, -1.385609512374507e-01, -1.366369481746035e-01, -5.994054791863528e-01, -6.076601488091260e-01, -1.162196696951506e-01, -1.156948212363630e-01, -1.179350111758385e-01, -1.425653888795294e-01, -1.179350111758384e-01, -1.425653888795293e-01, -2.558116491839646e-04, -3.053355110003709e-04, -3.054063376758114e-04, -3.745976638463132e-04, -4.215092050930094e-02, -4.778755966203131e-02, -4.871936624116032e-05, -4.570745051792792e-05, -1.737278787574284e-04, -5.386538542015526e-05, -1.737278787574285e-04, -5.386538542015488e-05, -3.416618895818035e+00, -3.415851853286799e+00, -3.417947737442108e+00, -3.417139062644459e+00, -3.416690142175513e+00, -3.415896181437027e+00, -3.417833023981241e+00, -3.417062875419374e+00, -3.417299869890178e+00, -3.416498854442464e+00, -3.417299869890178e+00, -3.416498854442464e+00, -1.274085394421209e+00, -1.273935017689928e+00, -1.283019603306007e+00, -1.282672578286578e+00, -1.265520982906255e+00, -1.267040176088692e+00, -1.273201855313840e+00, -1.274820862114575e+00, -1.284978641237101e+00, -1.280466761853920e+00, -1.284978641237101e+00, -1.280466761853920e+00, -4.232524823941346e-01, -4.223920895892904e-01, -4.618533498903752e-01, -4.617858218385185e-01, -3.953210535851232e-01, -3.997166959994702e-01, -4.138437452315666e-01, -4.197007412188599e-01, -4.346189683613943e-01, -4.244725183499789e-01, -4.346189683613942e-01, -4.244725183499790e-01, -1.158168169480365e-01, -1.144786895718501e-01, -1.416937464568985e-01, -1.415976859856159e-01, -1.147065473955678e-01, -1.125752839778577e-01, -1.320928357032395e+00, -1.320454138270791e+00, -1.126388856484837e-01, -1.011906720720463e-01, -1.126388856484837e-01, -1.011906720720463e-01, -2.385936187451852e-05, -2.480310983366148e-05, -4.792370454971759e-05, -4.808516942471625e-05, -4.916740281900110e-05, -6.156576090136184e-05, -9.778330491075414e-02, -9.841818137079499e-02, -3.772174002980832e-05, -9.466671541916864e-05, -3.772174002980633e-05, -9.466671541916692e-05, -4.279642644951587e-01, -4.289255679829436e-01, -4.243035160651787e-01, -4.252562479759054e-01, -4.257282865013957e-01, -4.266875045684582e-01, -4.268162344948405e-01, -4.277656075298703e-01, -4.262884855955614e-01, -4.272424370190146e-01, -4.262884855955614e-01, -4.272424370190146e-01, -4.173015601685426e-01, -4.181294725249531e-01, -3.505136249296670e-01, -3.511412657444896e-01, -3.691506174595640e-01, -3.699241372504185e-01, -3.887243323175931e-01, -3.894354151807435e-01, -3.787913482525058e-01, -3.794550845239172e-01, -3.787913482525058e-01, -3.794550845239172e-01, -4.835584583777565e-01, -4.836033673708586e-01, -1.687830415318277e-01, -1.686647597132290e-01, -1.972328406156742e-01, -1.970475326614757e-01, -2.594972781086848e-01, -2.600415084744900e-01, -2.247784378489638e-01, -2.246558362164093e-01, -2.247784378489638e-01, -2.246558362164093e-01, -3.277742498237071e-01, -3.282841534521846e-01, -3.878784284857566e-02, -3.945020010353593e-02, -7.351459983207168e-02, -7.758352249570633e-02, -2.536855902602599e-01, -2.525069099756992e-01, -1.051776790980409e-01, -9.765945064630023e-02, -1.051776790980409e-01, -9.765945064630020e-02, -7.004546376441918e-04, -7.888010653189681e-04, -1.581979686472155e-06, -2.132042965096255e-06, -1.114003648660798e-05, -1.376679560877766e-05, -1.013050755564333e-01, -1.013559311072292e-01, -4.589468711321417e-05, -7.367530737772984e-05, -4.589468711321290e-05, -7.367530737773068e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pwb6k_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pwb6k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.884291427413163e-09, 0.000000000000000e+00, -8.884380169432911e-09, -8.884256424199843e-09, 0.000000000000000e+00, -8.884354243525537e-09, -8.884019860741389e-09, 0.000000000000000e+00, -8.884052591889913e-09, -8.884424531271247e-09, 0.000000000000000e+00, -8.884560618977732e-09, -8.884281240921140e-09, 0.000000000000000e+00, -8.884263639266839e-09, -8.884281240921140e-09, 0.000000000000000e+00, -8.884263639266839e-09, -9.575525588797869e-06, 0.000000000000000e+00, -9.576577204625892e-06, -9.576051273100750e-06, 0.000000000000000e+00, -9.577045101519223e-06, -9.587526215479800e-06, 0.000000000000000e+00, -9.589696792137634e-06, -9.576096239593356e-06, 0.000000000000000e+00, -9.576778648997326e-06, -9.579618444819699e-06, 0.000000000000000e+00, -9.586856299209297e-06, -9.579618444819699e-06, 0.000000000000000e+00, -9.586856299209297e-06, -2.896581956231927e-03, 0.000000000000000e+00, -2.774006451859511e-03, -2.951684836607085e-03, 0.000000000000000e+00, -2.796815620831264e-03, -3.653133619620844e-03, 0.000000000000000e+00, -3.811560994896022e-03, -3.767366353522313e-03, 0.000000000000000e+00, -3.753023900086350e-03, -2.705037784961541e-03, 0.000000000000000e+00, -4.082935260234784e-03, -2.705037784961541e-03, 0.000000000000000e+00, -4.082935260234784e-03, -4.761451294301794e-01, 0.000000000000000e+00, -4.877516978810511e-01, -4.638488929748686e-01, 0.000000000000000e+00, -4.749629557002438e-01, -1.143823975000326e-04, 0.000000000000000e+00, 1.197756700100387e-04, -6.804010803065701e-01, 0.000000000000000e+00, -6.939654445652460e-01, -5.861687217100430e-01, 0.000000000000000e+00, 1.679761729806058e+00, -5.861687217100423e-01, 0.000000000000000e+00, 1.679761729806056e+00, 5.706260027743163e+01, 0.000000000000000e+00, 5.648995245902425e+01, 6.080783730192769e+01, 0.000000000000000e+00, 6.052807220699580e+01, 3.533737423033953e+01, 0.000000000000000e+00, 3.374784942225713e+01, 4.771228875765667e+01, 0.000000000000000e+00, 4.643421030898826e+01, 5.449499876757052e+01, 0.000000000000000e+00, 1.459102746799685e+02, 5.449499876756990e+01, 0.000000000000000e+00, 1.459102746799717e+02, -2.519395261151975e-06, 0.000000000000000e+00, -2.521561248043529e-06, -2.521602479137810e-06, 0.000000000000000e+00, -2.523690538738148e-06, -2.519511673027064e-06, 0.000000000000000e+00, -2.521626780797515e-06, -2.521400556799871e-06, 0.000000000000000e+00, -2.523565391780658e-06, -2.520532516357155e-06, 0.000000000000000e+00, -2.522630092844499e-06, -2.520532516357155e-06, 0.000000000000000e+00, -2.522630092844499e-06, -7.325761435074655e-05, 0.000000000000000e+00, -7.329663041613739e-05, -7.197011989413473e-05, 0.000000000000000e+00, -7.204543555828766e-05, -7.331658105862986e-05, 0.000000000000000e+00, -7.344125907865275e-05, -7.220974287376273e-05, 0.000000000000000e+00, -7.230388081746293e-05, -7.251240602000166e-05, 0.000000000000000e+00, -7.251882969174051e-05, -7.251240602000166e-05, 0.000000000000000e+00, -7.251882969174051e-05, -6.946568715342881e-03, 0.000000000000000e+00, -7.077075957114667e-03, -2.351763437418252e-03, 0.000000000000000e+00, -2.490474125226542e-03, -6.898268302955798e-03, 0.000000000000000e+00, -7.500501022663340e-03, 5.623233037127060e-03, 0.000000000000000e+00, 1.318000966070185e-03, -6.498703386610534e-03, 0.000000000000000e+00, -6.591428942439529e-03, -6.498703386610541e-03, 0.000000000000000e+00, -6.591428942439533e-03, 9.008874251918032e-01, 0.000000000000000e+00, 7.338268818491699e-01, -3.420438924884052e-01, 0.000000000000000e+00, -3.416382978471372e-01, 2.164479878662180e+00, 0.000000000000000e+00, 1.262307021460613e+00, -8.221102097495088e-05, 0.000000000000000e+00, -8.232098189273547e-05, -2.767601702158000e-01, 0.000000000000000e+00, -1.003835468305821e+00, -2.767601702158000e-01, 0.000000000000000e+00, -1.003835468305821e+00, 6.558042242675323e+01, 0.000000000000000e+00, 5.652093197304958e+01, 5.795203625607482e+01, 0.000000000000000e+00, 5.342377588011329e+01, 3.397547627914103e+02, 0.000000000000000e+00, 3.777325289242655e+02, 1.417162840846561e+01, 0.000000000000000e+00, 1.379498091945746e+01, 1.729507874257407e+02, 0.000000000000000e+00, 1.657078566400814e+02, 1.729507874257376e+02, 0.000000000000000e+00, 1.657078566400791e+02, -8.839038676561809e-03, 0.000000000000000e+00, -8.881981130690782e-03, -8.107178969907272e-03, 0.000000000000000e+00, -8.067351442814608e-03, -7.949442921804798e-03, 0.000000000000000e+00, -7.927718980199049e-03, -8.012800512938343e-03, 0.000000000000000e+00, -8.015461897649423e-03, -7.940746153305561e-03, 0.000000000000000e+00, -7.930656579437556e-03, -7.940746153305561e-03, 0.000000000000000e+00, -7.930656579437556e-03, -8.464527632771078e-03, 0.000000000000000e+00, -8.511980387260840e-03, -1.579387098489139e-02, 0.000000000000000e+00, -1.567473726444646e-02, -1.370798058957979e-02, 0.000000000000000e+00, -1.359747101544313e-02, -1.123207054109448e-02, 0.000000000000000e+00, -1.115319571352097e-02, -1.250950778090972e-02, 0.000000000000000e+00, -1.243706839472653e-02, -1.250950778090972e-02, 0.000000000000000e+00, -1.243706839472652e-02, -1.452136904262822e-03, 0.000000000000000e+00, -1.678499352154661e-03, -1.915700359397988e-01, 0.000000000000000e+00, -1.911192834272938e-01, -1.179332469511927e-01, 0.000000000000000e+00, -1.177983917158090e-01, -4.462720694363671e-02, 0.000000000000000e+00, -4.401790353915410e-02, -7.830403841622371e-02, 0.000000000000000e+00, -7.862602017516601e-02, -7.830403841622377e-02, 0.000000000000000e+00, -7.862602017516604e-02, -1.939522213124740e-02, 0.000000000000000e+00, -1.938075556797580e-02, 3.310256365157291e+01, 0.000000000000000e+00, 3.297080112583473e+01, 2.375352458939133e+01, 0.000000000000000e+00, 2.253440848757552e+01, -4.137511481863092e-02, 0.000000000000000e+00, -4.831583896332414e-02, 5.014317285750869e+00, 0.000000000000000e+00, 2.502385688219234e+00, 5.014317285750866e+00, 0.000000000000000e+00, 2.502385688219221e+00, 4.766782363415879e+01, 0.000000000000000e+00, 4.875030615849023e+01, 1.859664130269966e+02, 0.000000000000000e+00, 3.313691587457379e+02, 1.217176916830981e+02, 0.000000000000000e+00, 1.289603581109768e+02, 6.007185283484110e+00, 0.000000000000000e+00, 5.410569794225607e+00, 3.545962353155571e+02, 0.000000000000000e+00, 1.706341209887634e+02, 3.545962353155538e+02, 0.000000000000000e+00, 1.706341209887633e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pwb6k_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pwb6k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pwb6k_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pwb6k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.362622830026137e-05, -1.362629078144108e-05, -1.362653303443028e-05, -1.362651193650670e-05, -1.362697794738599e-05, -1.362729786967701e-05, -1.362355033966324e-05, -1.362292483675406e-05, -1.362643681267951e-05, -1.362428290315476e-05, -1.362643681267951e-05, -1.362428290315476e-05, -2.566667043412254e-04, -2.570002411473832e-04, -2.567062877839895e-04, -2.570616083480812e-04, -2.578887073464061e-04, -2.581971840470379e-04, -2.557712494808091e-04, -2.561238360368809e-04, -2.566123343889665e-04, -2.571538524514212e-04, -2.566123343889665e-04, -2.571538524514212e-04, -2.256144381485359e-03, -2.397187693484836e-03, -2.210737312021414e-03, -2.381098824071075e-03, -1.697652693905874e-03, -1.537517728675207e-03, -1.598543324909573e-03, -1.652067124812057e-03, -2.618959053840532e-03, -9.506052035610115e-04, -2.618959053840532e-03, -9.506052035610115e-04, -2.078147898410112e-03, -2.361108841714490e-03, -2.273778960684488e-03, -2.615919980784042e-03, -1.862614330937461e-03, -2.290531474118856e-03, -8.185572591357335e-04, -8.304126454236509e-04, -2.045604488961776e-03, -3.679835428281312e-04, -2.045604488961774e-03, -3.679835428281319e-04, -2.549661222532765e-07, -2.993554717290594e-07, -3.063833957660487e-07, -3.697434641016700e-07, -1.954559209819824e-05, -2.287308112979722e-05, -5.541274563228164e-08, -5.139372073583297e-08, -1.845882096751645e-07, -9.475341902089645e-08, -1.845882096751649e-07, -9.475341902089670e-08, -2.152592819606867e-04, -2.153020838955574e-04, -2.163707105803098e-04, -2.163763811526161e-04, -2.153152774274604e-04, -2.153349058900472e-04, -2.162689121420635e-04, -2.163108335614346e-04, -2.158322106958445e-04, -2.158417326453482e-04, -2.158322106958445e-04, -2.158417326453482e-04, -2.366129149686234e-04, -2.366122328775379e-04, -2.414820659587918e-04, -2.414058982378560e-04, -2.191198576687198e-04, -2.241747821014379e-04, -2.231782907925808e-04, -2.281839794886630e-04, -2.530258381143319e-04, -2.410527034000030e-04, -2.530258381143319e-04, -2.410527034000030e-04, -5.908372136377387e-03, -5.949198395404278e-03, -1.150944433991736e-02, -1.162547819963879e-02, -5.842122621358342e-03, -6.020495613947124e-03, -1.279815768845700e-02, -1.197825089529488e-02, -6.248715460168843e-03, -6.530121857106592e-03, -6.248715460168845e-03, -6.530121857106590e-03, -2.376983230515903e-04, -2.503006984532222e-04, -9.283131257057837e-04, -9.375707596865875e-04, -1.967780840736895e-04, -2.256059076279967e-04, -1.875646326205904e-03, -1.878139381732665e-03, -4.270814506885760e-04, -6.776598584468131e-04, -4.270814506885760e-04, -6.776598584468131e-04, -3.428896275630120e-08, -3.312775246136631e-08, -6.003266816228415e-08, -5.784400424278212e-08, -1.493273635742526e-07, -1.957510769144112e-07, -9.667507385477042e-05, -9.453874864988047e-05, -7.885843183743488e-08, -1.914015221014676e-07, -7.885843183743465e-08, -1.914015221014674e-07, -1.698571261031203e-02, -1.695101043933415e-02, -1.348632494679405e-02, -1.349478633618805e-02, -1.458240596509246e-02, -1.458612985616782e-02, -1.560230604158589e-02, -1.557855475385937e-02, -1.508014638923296e-02, -1.507028391803211e-02, -1.508014638923296e-02, -1.507028391803211e-02, -1.918781967510098e-02, -1.909730464318302e-02, -3.964318527272296e-03, -4.002430337436982e-03, -5.930785247862101e-03, -5.999443861149054e-03, -9.279800764276135e-03, -9.286952316560385e-03, -7.409243897450323e-03, -7.417579375785424e-03, -7.409243897450323e-03, -7.417579375785423e-03, -1.026459935741967e-02, -1.040654221584794e-02, -1.337069535182101e-03, -1.358127790277550e-03, -2.127394919626820e-03, -2.218808640699231e-03, -5.669996718828915e-03, -5.704229352336682e-03, -3.553222511479334e-03, -3.600763776356541e-03, -3.553222511479334e-03, -3.600763776356544e-03, -4.353506308868104e-03, -4.443959844507336e-03, -1.692461024064949e-05, -1.726782699983608e-05, -3.812141927297320e-05, -4.279735631023107e-05, -8.986410968639020e-03, -9.211137969575468e-03, -1.863950589122919e-04, -2.666581993754180e-04, -1.863950589122920e-04, -2.666581993754181e-04, -5.501477594032210e-07, -6.202438333463314e-07, -4.896332290937655e-09, -8.731230544710106e-09, -2.361461249432885e-08, -3.001610358517400e-08, -1.845708343368359e-04, -1.896228599547584e-04, -1.392531259664556e-07, -1.550835591712349e-07, -1.392531259664558e-07, -1.550835591712351e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05