
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_hjs_b88_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.058652718366064e+01, -2.058656283076210e+01, -2.058677427724535e+01, -2.058624282929382e+01, -2.058651376717025e+01, -2.058651376717025e+01, -3.344377711858403e+00, -3.344353144031911e+00, -3.343868712361173e+00, -3.345357941430168e+00, -3.344430619156008e+00, -3.344430619156008e+00, -6.147686858179082e-01, -6.145806555725180e-01, -6.114881923992868e-01, -6.162886050865706e-01, -6.168959905489730e-01, -6.168959905489730e-01, -1.434042828298973e-01, -1.444564741720915e-01, -7.216442153630791e-01, -1.168227784062663e-01, -1.408204949950067e-01, -1.408204949950065e-01, -3.308076931058759e-02, -3.091248484054290e-02, -4.335511515430356e-02, 0.000000000000000e+00, -2.761036699598476e-02, -2.761036699598490e-02, -4.920835941894431e+00, -4.920644721342471e+00, -4.920835824093220e+00, -4.920666876454451e+00, -4.920734266795554e+00, -4.920734266795554e+00, -1.968413869506041e+00, -1.979105551886227e+00, -1.966925378014147e+00, -1.976345454928470e+00, -1.975087662567397e+00, -1.975087662567397e+00, -5.099531416810833e-01, -5.397346658108760e-01, -4.706867557262037e-01, -4.739987378095185e-01, -5.173470273115228e-01, -5.173470273115228e-01, -9.107620975721553e-02, -1.598684975343961e-01, -8.479959961322851e-02, -1.744259696999614e+00, -9.657062126659383e-02, -9.657062126659383e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.876844324191321e-02, 0.000000000000000e+00, 0.000000000000000e+00, -4.901410005440282e-01, -4.913848262316288e-01, -4.909630962380511e-01, -4.905970154360782e-01, -4.907808860740999e-01, -4.907808860740999e-01, -4.740133343542223e-01, -4.343229935348025e-01, -4.455665394878888e-01, -4.565675127605987e-01, -4.508168194729898e-01, -4.508168194729898e-01, -5.694580476638930e-01, -1.994731997589991e-01, -2.328137786303656e-01, -2.930585571581451e-01, -2.598879428131904e-01, -2.598879428131903e-01, -3.959451019271748e-01, -4.327002041107936e-02, -5.367443236689012e-02, -2.734277216367494e-01, -6.779944209477294e-02, -6.779944209477286e-02, -7.992756557977736e-05, 0.000000000000000e+00, 0.000000000000000e+00, -6.462711034092070e-02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_hjs_b88_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.549830332281552e+01, -2.549827214818754e+01, -2.549840770933328e+01, -2.549834796874958e+01, -2.549872879524717e+01, -2.549881166423844e+01, -2.549755026323594e+01, -2.549730941915793e+01, -2.549836623087831e+01, -2.549790044324786e+01, -2.549836623087831e+01, -2.549790044324786e+01, -4.062875725244372e+00, -4.062979073049798e+00, -4.062908608971092e+00, -4.063014134291792e+00, -4.063728507743809e+00, -4.063967594752341e+00, -4.062973323406083e+00, -4.063206779922125e+00, -4.062123104690061e+00, -4.064011561956576e+00, -4.062123104690061e+00, -4.064011561956576e+00, -7.182744881839023e-01, -7.230910342989874e-01, -7.164753820145787e-01, -7.223778346184810e-01, -6.928245191781714e-01, -6.849579030947350e-01, -6.943069432806598e-01, -6.968420690223684e-01, -7.317064077178944e-01, -6.560397840954906e-01, -7.317064077178944e-01, -6.560397840954906e-01, -1.280650776282327e-01, -1.359650221756440e-01, -1.307801902566097e-01, -1.400047615325304e-01, -8.349209182787032e-01, -8.728413311318518e-01, -8.424503552671198e-02, -8.610925419153162e-02, -1.362595018599675e-01, -4.921158781994682e-02, -1.362595018599672e-01, -4.921158781994794e-02, 1.163773217261530e-01, 6.981191831447034e-02, 6.733649693209109e-02, 1.065584153477918e-02, 1.176492972240748e-04, -1.176488836541368e-03, 0.000000000000000e+00, 0.000000000000000e+00, 3.061463701936082e-01, -4.116442787938041e-18, 3.061463701936118e-01, 4.871041733077557e-19, -6.214456158587927e+00, -6.212910009936518e+00, -6.216681868124058e+00, -6.215060145969626e+00, -6.214581378666103e+00, -6.212985023557582e+00, -6.216491135380603e+00, -6.214938298495608e+00, -6.215594845639554e+00, -6.213989624104397e+00, -6.215594845639554e+00, -6.213989624104397e+00, -2.182257800346876e+00, -2.182138193133799e+00, -2.201108108264586e+00, -2.200436971508324e+00, -2.162417295885791e+00, -2.168007822059964e+00, -2.178660061939310e+00, -2.184435887584297e+00, -2.207722694655099e+00, -2.193802285370632e+00, -2.207722694655099e+00, -2.193802285370632e+00, -6.400363870451361e-01, -6.382481415704117e-01, -7.181934709590725e-01, -7.188058875840933e-01, -5.719459647487991e-01, -5.968827858810649e-01, -6.155937978993083e-01, -6.376588335337241e-01, -6.706680773192571e-01, -6.335680479214467e-01, -6.706680773192569e-01, -6.335680479214468e-01, -4.586888199284078e-02, -4.659396888736592e-02, -1.290691281864610e-01, -1.300132878397027e-01, -3.798736250443489e-02, -4.189108296709775e-02, -2.284743695396998e+00, -2.283732541375927e+00, -5.765932909743803e-02, -6.346655770501979e-02, -5.765932909743803e-02, -6.346655770501979e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.612346864932623e-02, -1.646677349781482e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -6.644891498526537e-01, -6.673756888137107e-01, -6.550746118804914e-01, -6.580104241654390e-01, -6.583133262576178e-01, -6.612534267592466e-01, -6.610679613540629e-01, -6.639533320107279e-01, -6.596846367146808e-01, -6.625966316053663e-01, -6.596846367146808e-01, -6.625966316053663e-01, -6.467469418915054e-01, -6.490909864422720e-01, -5.130795555906584e-01, -5.158680996547482e-01, -5.493743038903758e-01, -5.524258806758986e-01, -5.871679481158230e-01, -5.895605502999336e-01, -5.679313381987166e-01, -5.703899176863473e-01, -5.679313381987166e-01, -5.703899176863473e-01, -7.542767778671797e-01, -7.560171359671241e-01, -1.799726275356374e-01, -1.814495375160537e-01, -2.320591079410183e-01, -2.354745622642473e-01, -3.430745555289656e-01, -3.454794868008941e-01, -2.841964582220720e-01, -2.842172578175975e-01, -2.841964582220733e-01, -2.842172578175968e-01, -4.665030491245672e-01, -4.706710174468915e-01, 4.804346493326253e-04, 3.252863422784539e-04, -7.737016251696456e-03, -8.812829427605378e-03, -3.323335262202368e-01, -3.387852936089096e-01, -2.704846112158076e-02, -3.000844178231531e-02, -2.704846112158048e-02, -3.000844178231497e-02, -1.510157132616658e-04, -1.680752353222921e-04, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.441605344319109e-02, -2.535443677431552e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_hjs_b88_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.981001672392621e-09, 0.000000000000000e+00, -9.981057966878486e-09, -9.980933910671299e-09, 0.000000000000000e+00, -9.981008704037164e-09, -9.980609807909569e-09, 0.000000000000000e+00, -9.980574338108714e-09, -9.981377822162537e-09, 0.000000000000000e+00, -9.981552382255311e-09, -9.980966677507222e-09, 0.000000000000000e+00, -9.981082454838091e-09, -9.980966677507222e-09, 0.000000000000000e+00, -9.981082454838091e-09, -1.334204123723643e-05, 0.000000000000000e+00, -1.334633350217527e-05, -1.334232059462570e-05, 0.000000000000000e+00, -1.334694337600154e-05, -1.335244173353503e-05, 0.000000000000000e+00, -1.335478877194790e-05, -1.332630670662359e-05, 0.000000000000000e+00, -1.332947140291913e-05, -1.334956239469888e-05, 0.000000000000000e+00, -1.333730888448495e-05, -1.334956239469888e-05, 0.000000000000000e+00, -1.333730888448495e-05, -8.518251243003306e-03, 0.000000000000000e+00, -8.490279077746510e-03, -8.534865729545604e-03, 0.000000000000000e+00, -8.500103273452500e-03, -8.773310492371303e-03, 0.000000000000000e+00, -8.826323152910841e-03, -8.553460571024057e-03, 0.000000000000000e+00, -8.540297976605635e-03, -8.392016955617010e-03, 0.000000000000000e+00, -8.709093414024387e-03, -8.392016955617010e-03, 0.000000000000000e+00, -8.709093414024387e-03, -1.214825133951909e+00, 0.000000000000000e+00, -1.090016182283217e+00, -1.186131148766672e+00, 0.000000000000000e+00, -1.045986253836471e+00, -4.938471105519598e-03, 0.000000000000000e+00, -4.548912523702126e-03, -2.459782340324651e+00, 0.000000000000000e+00, -2.352647856242507e+00, -1.037364797733161e+00, 0.000000000000000e+00, -6.034613286178335e+00, -1.037364797733129e+00, 0.000000000000000e+00, -6.034613286178321e+00, -5.705529093976069e+04, 0.000000000000000e+00, -3.384981499285013e+04, -3.520872134452419e+04, 0.000000000000000e+00, -1.515784143104292e+04, -9.216169576575459e+01, 0.000000000000000e+00, -7.988889699166621e+01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.760201027268116e+05, 0.000000000000000e+00, 0.000000000000000e+00, -1.760201027268153e+05, 0.000000000000000e+00, 0.000000000000000e+00, -2.976599189408850e-06, 0.000000000000000e+00, -2.979305546442884e-06, -2.977995916744120e-06, 0.000000000000000e+00, -2.980657238891410e-06, -2.976648819223591e-06, 0.000000000000000e+00, -2.979331981502289e-06, -2.977847560329132e-06, 0.000000000000000e+00, -2.980560149419306e-06, -2.977335907225541e-06, 0.000000000000000e+00, -2.979987795700413e-06, -2.977335907225541e-06, 0.000000000000000e+00, -2.979987795700413e-06, -1.072844417438121e-04, 0.000000000000000e+00, -1.073041615227550e-04, -1.048854362783213e-04, 0.000000000000000e+00, -1.049829668323432e-04, -1.080005391068310e-04, 0.000000000000000e+00, -1.078361344288509e-04, -1.059223792998700e-04, 0.000000000000000e+00, -1.057155405979157e-04, -1.053544783789936e-04, 0.000000000000000e+00, -1.060085552582529e-04, -1.053544783789936e-04, 0.000000000000000e+00, -1.060085552582529e-04, -1.638967980940545e-02, 0.000000000000000e+00, -1.656928909488117e-02, -1.369900021771411e-02, 0.000000000000000e+00, -1.369864921223890e-02, -2.333178078889145e-02, 0.000000000000000e+00, -2.056550611707285e-02, -2.303887533703894e-02, 0.000000000000000e+00, -2.024097477864089e-02, -1.435284625565947e-02, 0.000000000000000e+00, -1.742489058246995e-02, -1.435284625565945e-02, 0.000000000000000e+00, -1.742489058246998e-02, -6.294580635352269e+00, 0.000000000000000e+00, -6.176767536711294e+00, -9.380282614259848e-01, 0.000000000000000e+00, -9.238541114618546e-01, -8.460282321495724e+00, 0.000000000000000e+00, -7.326807091632288e+00, -1.762381331251777e-04, 0.000000000000000e+00, -1.765744144012205e-04, -4.611969081311151e+00, 0.000000000000000e+00, -4.255441127069346e+00, -4.611969081311151e+00, 0.000000000000000e+00, -4.255441127069346e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.502181789772473e+01, 0.000000000000000e+00, -2.436583572582879e+01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.945072879866001e-02, 0.000000000000000e+00, -1.917329704113418e-02, -1.917737096782469e-02, 0.000000000000000e+00, -1.890927132754710e-02, -1.926966220218928e-02, 0.000000000000000e+00, -1.899906371718843e-02, -1.935040400710079e-02, 0.000000000000000e+00, -1.907583934300874e-02, -1.930982627580114e-02, 0.000000000000000e+00, -1.903719465220462e-02, -1.930982627580114e-02, 0.000000000000000e+00, -1.903719465220462e-02, -2.186155955649128e-02, 0.000000000000000e+00, -2.158548481600080e-02, -2.934331066140977e-02, 0.000000000000000e+00, -2.893013001588678e-02, -2.660181132564476e-02, 0.000000000000000e+00, -2.622876828750148e-02, -2.449490739824689e-02, 0.000000000000000e+00, -2.418797238537405e-02, -2.553322962682768e-02, 0.000000000000000e+00, -2.519633249434740e-02, -2.553322962682768e-02, 0.000000000000000e+00, -2.519633249434740e-02, -1.134360833415717e-02, 0.000000000000000e+00, -1.130228090105300e-02, -4.448576169422141e-01, 0.000000000000000e+00, -4.373503802364637e-01, -2.568441720502869e-01, 0.000000000000000e+00, -2.498318226131052e-01, -1.115576917336109e-01, 0.000000000000000e+00, -1.094402291054763e-01, -1.691140804592924e-01, 0.000000000000000e+00, -1.697629580053046e-01, -1.691140804592907e-01, 0.000000000000000e+00, -1.697629580053057e-01, -4.056474758869077e-02, 0.000000000000000e+00, -3.969660059168299e-02, -9.479878346242999e+01, 0.000000000000000e+00, -9.312415765133321e+01, -4.301326943175282e+01, 0.000000000000000e+00, -3.965148937420443e+01, -1.391675574579018e-01, 0.000000000000000e+00, -1.323058066219183e-01, -1.425393993011312e+01, 0.000000000000000e+00, -1.301137327316511e+01, -1.425393993011324e+01, 0.000000000000000e+00, -1.301137327316510e+01, -8.769922039436323e-04, 0.000000000000000e+00, -1.023415120089714e-03, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.644857078250179e+01, 0.000000000000000e+00, -1.567643704425708e+01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05