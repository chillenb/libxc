
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revtpss_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.257942688191871e+01, -2.257949458166019e+01, -2.257991913724681e+01, -2.257891546419148e+01, -2.257942366833433e+01, -2.257942366833433e+01, -3.397869483574304e+00, -3.397883183611698e+00, -3.398477814734111e+00, -3.400203109058449e+00, -3.398973915318369e+00, -3.398973915318369e+00, -6.550192304371378e-01, -6.546834843742060e-01, -6.483889345253298e-01, -6.548623522976771e-01, -6.563260440985050e-01, -6.563260440985050e-01, -1.968920768654093e-01, -1.981981043398438e-01, -7.550012502656139e-01, -1.670953539987898e-01, -1.926323863463194e-01, -1.926323863463193e-01, -1.010238653398351e-02, -1.063723306642128e-02, -5.724065924634543e-02, -5.828433494484976e-03, -8.132571873727011e-03, -8.132571873727013e-03, -5.551649108927802e+00, -5.552745059722679e+00, -5.551806909714986e+00, -5.552770582306875e+00, -5.552149367703282e+00, -5.552149367703282e+00, -2.062622775843251e+00, -2.080104403049567e+00, -2.057943797192992e+00, -2.073413770623960e+00, -2.074570525554558e+00, -2.074570525554558e+00, -5.902764088153377e-01, -6.103560322738303e-01, -5.357268501285704e-01, -5.345665721751256e-01, -6.000155190850645e-01, -6.000155190850645e-01, -1.323910482308077e-01, -2.108199952892553e-01, -1.244002285367344e-01, -1.809138287377219e+00, -1.445979199592598e-01, -1.445979199592598e-01, -4.500150006913003e-03, -5.697585417076378e-03, -4.362630379116023e-03, -8.862887530244774e-02, -5.479896711911908e-03, -5.479896711911909e-03, -5.850211756581325e-01, -6.040141658985638e-01, -6.000552214543249e-01, -5.949857803508916e-01, -5.977651233874131e-01, -5.977651233874131e-01, -5.465667131272471e-01, -5.209837007849357e-01, -5.429381786981208e-01, -5.609242728577031e-01, -5.520709158596884e-01, -5.520709158596884e-01, -6.370737389595871e-01, -2.515933790824294e-01, -2.873592171177342e-01, -3.551707196649148e-01, -3.191410366216017e-01, -3.191410366216016e-01, -4.691384365723901e-01, -5.488622134861036e-02, -7.346276548524169e-02, -3.397437529871133e-01, -1.073974665780573e-01, -1.073974665780573e-01, -1.422968302533421e-02, -1.523238629952538e-03, -3.203047975489215e-03, -1.021938891228247e-01, -5.032581815720692e-03, -5.032581815720688e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revtpss_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.717726682966539e+01, -2.717825911630662e+01, -2.717733730325653e+01, -2.717831020856588e+01, -2.717775024110452e+01, -2.717884943512890e+01, -2.717695963250396e+01, -2.717784034481597e+01, -2.717730334039316e+01, -2.717838090900117e+01, -2.717730334039316e+01, -2.717838090900117e+01, -4.213204430759285e+00, -4.213253673613075e+00, -4.213237511246806e+00, -4.213283617942484e+00, -4.213998760161837e+00, -4.214212908765890e+00, -4.213527582858137e+00, -4.213705399183672e+00, -4.212442742777264e+00, -4.214463149061651e+00, -4.212442742777264e+00, -4.214463149061651e+00, -7.946458719703978e-01, -8.005207499701017e-01, -7.924365149688160e-01, -7.996998813808573e-01, -7.624237386061985e-01, -7.518954183962382e-01, -7.631485137295675e-01, -7.665448905987681e-01, -8.109463448118551e-01, -7.135174314191076e-01, -8.109463448118551e-01, -7.135174314191076e-01, -1.860970939285832e-01, -1.978063776663669e-01, -1.888762601164154e-01, -2.027367701318865e-01, -9.284637374846605e-01, -9.719662315395216e-01, -1.472549510332179e-01, -1.497459655997853e-01, -1.941569140262197e-01, -1.302159202913773e-01, -1.941569140262196e-01, -1.302159202913773e-01, -1.300224646853699e-02, -1.381178914843110e-02, -1.362009481997162e-02, -1.458409481714142e-02, -7.078514287656158e-02, -7.412806467125936e-02, -7.829915754434594e-03, -7.699780992504404e-03, -1.160570367067287e-02, -6.611287353058101e-03, -1.160570367067287e-02, -6.611287353058102e-03, -6.628270230064671e+00, -6.627120163072497e+00, -6.628006789882702e+00, -6.626866298289124e+00, -6.628418606636332e+00, -6.627228899528022e+00, -6.628187052742123e+00, -6.626992847515258e+00, -6.628011965771454e+00, -6.626972343145444e+00, -6.628011965771454e+00, -6.626972343145444e+00, -2.437550870119238e+00, -2.439623336319348e+00, -2.469195062567894e+00, -2.470276972965887e+00, -2.423160077103510e+00, -2.429264584866151e+00, -2.451539032080629e+00, -2.457981610026515e+00, -2.465376064579408e+00, -2.457268814744985e+00, -2.465376064579408e+00, -2.457268814744985e+00, -7.182259600316551e-01, -7.169082611712694e-01, -7.921632455597022e-01, -7.926143439740013e-01, -6.455297387019782e-01, -6.714561493345678e-01, -6.874213986531563e-01, -7.109383688833940e-01, -7.489840227893498e-01, -7.087784879482655e-01, -7.489840227893501e-01, -7.087784879482655e-01, -1.352911166322643e-01, -1.351879321428287e-01, -1.894979288242286e-01, -1.902085955343334e-01, -1.285682040398648e-01, -1.313591553993484e-01, -2.389894488804701e+00, -2.388860617707155e+00, -1.365390946583390e-01, -1.328742832429731e-01, -1.365390946583390e-01, -1.328742832429731e-01, -5.875759873905722e-03, -6.106319756627619e-03, -7.533632012331734e-03, -7.647725651047965e-03, -5.631646124324437e-03, -5.962719895397889e-03, -1.025723844230482e-01, -1.033444599098976e-01, -5.753811141044111e-03, -7.895380595547896e-03, -5.753811141044111e-03, -7.895380595547898e-03, -7.282358416487414e-01, -7.311493380323835e-01, -7.211612477335073e-01, -7.239948814023168e-01, -7.217316550125089e-01, -7.245952587025343e-01, -7.234092729130656e-01, -7.262648785940269e-01, -7.223860887228083e-01, -7.252495786448897e-01, -7.223860887228083e-01, -7.252495786448898e-01, -7.179003619515069e-01, -7.201999363176664e-01, -6.334844638803969e-01, -6.363750048632186e-01, -6.572810551970594e-01, -6.600191104079527e-01, -6.711332811640122e-01, -6.733804802788200e-01, -6.640331612963439e-01, -6.667637176184921e-01, -6.640331612963439e-01, -6.667637176184921e-01, -8.300191230550497e-01, -8.316003040261256e-01, -2.408369772100470e-01, -2.424443093205929e-01, -3.006648570151633e-01, -3.051810348520680e-01, -4.194827074434455e-01, -4.216269652924083e-01, -3.633966907235254e-01, -3.635014211585407e-01, -3.633966907235253e-01, -3.635014211585407e-01, -5.593246962917586e-01, -5.653127006919301e-01, -6.985023273450230e-02, -7.026260224214827e-02, -8.922121031834167e-02, -9.135799054599698e-02, -4.033157577929758e-01, -4.148281665254272e-01, -1.142750304229462e-01, -1.135233424665577e-01, -1.142750304229462e-01, -1.135233424665576e-01, -1.855749566188168e-02, -1.922220029260836e-02, -2.028496636204756e-03, -2.033043268322707e-03, -4.126304985203939e-03, -4.388346295195118e-03, -1.105058926094659e-01, -1.116035817771567e-01, -5.448606474392077e-03, -7.238527494298673e-03, -5.448606474392071e-03, -7.238527494298666e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtpss_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.013137042645100e-08, 0.000000000000000e+00, -4.013375192592089e-08, -4.013252499019167e-08, 0.000000000000000e+00, -4.013458969183207e-08, -4.013417081940948e-08, 0.000000000000000e+00, -4.013752183856711e-08, -4.012120657746545e-08, 0.000000000000000e+00, -4.012097150733250e-08, -4.013217028152090e-08, 0.000000000000000e+00, -4.012607711721121e-08, -4.013217028152090e-08, 0.000000000000000e+00, -4.012607711721121e-08, -2.233401456734022e-05, 0.000000000000000e+00, -2.231164954929465e-05, -2.234205272577266e-05, 0.000000000000000e+00, -2.231531538473464e-05, -2.248026060305216e-05, 0.000000000000000e+00, -2.249946055098151e-05, -2.249836216520085e-05, 0.000000000000000e+00, -2.248092677004015e-05, -2.233217420315063e-05, 0.000000000000000e+00, -2.260566787946820e-05, -2.233217420315063e-05, 0.000000000000000e+00, -2.260566787946820e-05, -7.887374511074685e-03, 0.000000000000000e+00, -7.706642488243499e-03, -7.967470594294583e-03, 0.000000000000000e+00, -7.725500634518173e-03, -8.833588951497346e-03, 0.000000000000000e+00, -9.236913784712893e-03, -9.223090262680816e-03, 0.000000000000000e+00, -9.215774903686252e-03, -7.126020280850868e-03, 0.000000000000000e+00, -1.014977786395217e-02, -7.126020280850868e-03, 0.000000000000000e+00, -1.014977786395217e-02, -1.300820105622964e+00, 0.000000000000000e+00, -1.153817736953463e+00, -1.290293874226232e+00, 0.000000000000000e+00, -1.123462011270910e+00, -2.489698986187642e-03, 0.000000000000000e+00, -1.935520881420034e-03, -2.133949676790412e+00, 0.000000000000000e+00, -2.043407497675403e+00, -1.118528361339571e+00, 0.000000000000000e+00, -3.392275910431577e+00, -1.118528361339570e+00, 0.000000000000000e+00, -3.392275910431577e+00, -7.681537599429603e+00, 0.000000000000000e+00, -7.604134898472878e+00, -8.093185988202542e+00, 0.000000000000000e+00, -8.051216353637477e+00, -4.433102185697854e+00, 0.000000000000000e+00, -4.450026854786140e+00, -7.037943886390873e+00, 0.000000000000000e+00, -6.846938520670945e+00, -7.669370438881876e+00, 0.000000000000000e+00, -1.946266578719472e+01, -7.669370438881907e+00, 0.000000000000000e+00, -1.946266578719473e+01, -1.694448351852847e-05, 0.000000000000000e+00, -1.695638917183131e-05, -1.709830024013992e-05, 0.000000000000000e+00, -1.710514200712312e-05, -1.695311980415825e-05, 0.000000000000000e+00, -1.696157673698037e-05, -1.708506317657297e-05, 0.000000000000000e+00, -1.709667412787444e-05, -1.702290408032961e-05, 0.000000000000000e+00, -1.703083995101330e-05, -1.702290408032961e-05, 0.000000000000000e+00, -1.703083995101330e-05, -1.463462872906149e-04, 0.000000000000000e+00, -1.460832691645249e-04, -1.438015831805059e-04, 0.000000000000000e+00, -1.436378435238989e-04, -1.408130898484627e-04, 0.000000000000000e+00, -1.421759534841683e-04, -1.383530467359832e-04, 0.000000000000000e+00, -1.396428718080253e-04, -1.492530519065594e-04, 0.000000000000000e+00, -1.454013766552747e-04, -1.492530519065594e-04, 0.000000000000000e+00, -1.454013766552747e-04, -4.667581712530985e-02, 0.000000000000000e+00, -4.800687324245897e-02, -4.217105791569533e-02, 0.000000000000000e+00, -4.401108997109288e-02, -4.324657347905530e-02, 0.000000000000000e+00, -4.640247641494361e-02, -3.013568946307913e-02, 0.000000000000000e+00, -3.392590406141847e-02, -4.769423253394257e-02, 0.000000000000000e+00, -5.091643550748084e-02, -4.769423253394256e-02, 0.000000000000000e+00, -5.091643550748082e-02, -2.779698552979664e+00, 0.000000000000000e+00, -2.815399700698409e+00, -8.928190366427873e-01, 0.000000000000000e+00, -8.826325263870450e-01, -3.191454682469149e+00, 0.000000000000000e+00, -3.062878845738068e+00, -2.408763231616053e-04, 0.000000000000000e+00, -2.413719270058630e-04, -2.870442346550933e+00, 0.000000000000000e+00, -3.245203734370107e+00, -2.870442346550933e+00, 0.000000000000000e+00, -3.245203734370107e+00, -9.880448767665097e+00, 0.000000000000000e+00, -8.551870557237446e+00, -8.507434027899317e+00, 0.000000000000000e+00, -7.854727810108486e+00, -4.845681097010100e+01, 0.000000000000000e+00, -5.390308042378994e+01, -4.934598600964802e+00, 0.000000000000000e+00, -4.754383784430670e+00, -2.409794037130614e+01, 0.000000000000000e+00, -2.366732084628480e+01, -2.409794037130607e+01, 0.000000000000000e+00, -2.366732084628480e+01, -3.822841936253505e-01, 0.000000000000000e+00, -3.866596593508253e-01, -2.270133839263154e-01, 0.000000000000000e+00, -2.283072047770711e-01, -2.703090406595035e-01, 0.000000000000000e+00, -2.725743532096617e-01, -3.166265614775544e-01, 0.000000000000000e+00, -3.189281634876180e-01, -2.921909139613769e-01, 0.000000000000000e+00, -2.944727282886953e-01, -2.921909139613769e-01, 0.000000000000000e+00, -2.944727282886953e-01, -2.081682865081790e-01, 0.000000000000000e+00, -2.142148335317048e-01, -5.295082797686073e-02, 0.000000000000000e+00, -5.285261551752193e-02, -7.379177214514918e-02, 0.000000000000000e+00, -7.412698052835461e-02, -1.213595680378490e-01, 0.000000000000000e+00, -1.208413509212753e-01, -9.303275609935426e-02, 0.000000000000000e+00, -9.261448178091566e-02, -9.303275609935426e-02, 0.000000000000000e+00, -9.261448178091566e-02, -2.684543329002027e-02, 0.000000000000000e+00, -2.870358820628634e-02, -4.579690701764420e-01, 0.000000000000000e+00, -4.513582438772712e-01, -2.864034533040482e-01, 0.000000000000000e+00, -2.802497146293279e-01, -1.859584632421962e-01, 0.000000000000000e+00, -1.831823486612514e-01, -2.181192486574466e-01, 0.000000000000000e+00, -2.198121923334152e-01, -2.181192486574466e-01, 0.000000000000000e+00, -2.198121923334153e-01, -7.455985864428388e-02, 0.000000000000000e+00, -7.451431846096977e-02, -4.059678019980007e+00, 0.000000000000000e+00, -4.063676520378008e+00, -3.953234820343543e+00, 0.000000000000000e+00, -4.029656370653010e+00, -2.788648471987217e-01, 0.000000000000000e+00, -2.979414954155080e-01, -4.749405574630917e+00, 0.000000000000000e+00, -5.436303443160932e+00, -4.749405574630920e+00, 0.000000000000000e+00, -5.436303443160935e+00, -6.085257687049814e+00, 0.000000000000000e+00, -6.211777748608748e+00, -3.040271095766849e+01, 0.000000000000000e+00, -5.385339430935281e+01, -1.876316473542080e+01, 0.000000000000000e+00, -1.997441687587944e+01, -5.314758420358999e+00, 0.000000000000000e+00, -5.193553325030436e+00, -4.968065908949434e+01, 0.000000000000000e+00, -2.457069557264073e+01, -4.968065908949445e+01, 0.000000000000000e+00, -2.457069557264080e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtpss_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtpss_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.929652666765388e-03, 2.930647702095269e-03, 2.929767909080669e-03, 2.930731284515703e-03, 2.930108213904239e-03, 2.931227032128473e-03, 2.928817254325320e-03, 2.929577732406921e-03, 2.929726598379635e-03, 2.930215146647812e-03, 2.929726598379635e-03, 2.930215146647812e-03, 2.862159216077433e-03, 2.853904862760215e-03, 2.864009695314134e-03, 2.854448768816547e-03, 2.892429342777986e-03, 2.895598333025929e-03, 2.913011677612756e-03, 2.906241712458087e-03, 2.859204390724733e-03, 2.932580604420950e-03, 2.859204390724733e-03, 2.932580604420950e-03, 3.250223655320498e-03, 3.155657440803606e-03, 3.297355892344769e-03, 3.157316740279834e-03, 3.600276126384356e-03, 3.893857491908255e-03, 4.478629735222313e-03, 4.568541428121943e-03, 2.691098804635415e-03, 4.906712075403674e-03, 2.691098804635415e-03, 4.906712075403674e-03, 9.957507012207181e-03, 1.237727713348749e-02, 1.106906227593308e-02, 1.415468067882831e-02, 1.967767010454460e-04, 1.322390682062589e-04, 1.467096315051056e-03, 1.879995153594644e-03, 9.971030462043118e-03, 1.961236507705284e-04, 9.971030462043113e-03, 1.961236507705288e-04, 3.037154967199000e-12, 5.354579289753514e-12, 4.594009590752744e-12, 7.614830355833938e-12, 2.317998455664730e-07, 3.508272222362653e-07, 6.721581582331462e-14, 5.722335851395381e-14, 1.320016539387003e-12, 1.423814029693235e-13, 1.320016539387011e-12, 1.423814029693235e-13, 1.902042865439206e-02, 1.902875764063136e-02, 1.919920865131873e-02, 1.920164896168664e-02, 1.903389695618339e-02, 1.903725939874394e-02, 1.918716758247299e-02, 1.919418531900720e-02, 1.910895419284903e-02, 1.911484826467657e-02, 1.910895419284903e-02, 1.911484826467657e-02, 6.142427234640114e-03, 6.166357107308272e-03, 6.415786776246084e-03, 6.431842415603447e-03, 5.758879825770961e-03, 5.891062428963553e-03, 5.980430606457645e-03, 6.118314505281508e-03, 6.590508961297246e-03, 6.349507671543748e-03, 6.590508961297246e-03, 6.349507671543748e-03, 4.437284228792009e-02, 4.613784303617157e-02, 2.946663317214544e-02, 3.111443270747839e-02, 2.171135810808247e-02, 3.019743656216546e-02, 8.692683604854973e-03, 1.344769500433797e-02, 5.410048520630156e-02, 4.434734491927683e-02, 5.410048520630156e-02, 4.434734491927683e-02, 1.306124011132199e-04, 1.420203787411045e-04, 3.617329681159997e-03, 3.672784695808329e-03, 7.558282634468624e-05, 1.097831780415353e-04, 2.958867071540816e-03, 2.960343180463543e-03, 4.860048366162074e-04, 1.228269912293091e-03, 4.860048366162074e-04, 1.228269912293092e-03, 1.132322489583667e-14, 1.177760588182113e-14, 6.685610262899771e-14, 6.149241892122686e-14, 3.172805700204245e-13, 5.839783670317079e-13, 1.084645700641510e-05, 1.152070427636040e-05, 4.095957694229130e-14, 8.478191120970510e-13, 4.095957694229121e-14, 8.478191120970499e-13, 3.454376296608642e-01, 3.532194993220177e-01, 2.773733721374231e-01, 2.827068604207122e-01, 3.123631727204844e-01, 3.191143157020560e-01, 3.396480928918140e-01, 3.467756650842463e-01, 3.267020128540338e-01, 3.336697203669538e-01, 3.267020128540338e-01, 3.336697203669536e-01, 7.801876904639536e-02, 8.235865375169625e-02, 4.614435312247029e-02, 4.680324993119606e-02, 7.311719738562515e-02, 7.447262066997802e-02, 1.238408462298669e-01, 1.245723984282585e-01, 9.373877183142706e-02, 9.477154486024657e-02, 9.373877183142702e-02, 9.477154486024653e-02, 1.961168216287087e-02, 2.156032104200824e-02, 7.392592915604130e-03, 7.573301954290608e-03, 1.436119397261489e-02, 1.516481531731743e-02, 3.414770406673969e-02, 3.396283616121447e-02, 2.571683193204924e-02, 2.603050288427639e-02, 2.571683193204922e-02, 2.603050288427641e-02, 3.778511136051337e-02, 3.981265716732521e-02, 1.843104951153788e-07, 1.891989062345788e-07, 1.335733180732568e-06, 1.775668189294368e-06, 4.360492871393340e-02, 5.873445547926312e-02, 5.735500288710572e-05, 1.219505451628167e-04, 5.735500288710579e-05, 1.219505451628170e-04, 1.906047166607047e-11, 2.490885837296794e-11, 2.585043439468274e-17, 3.358889468662572e-17, 5.546347178678012e-15, 9.669373239743592e-15, 5.460337016469947e-05, 7.853297260583738e-05, 1.537202316344564e-13, 4.960820056561384e-13, 1.537202316344560e-13, 4.960820056561375e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05