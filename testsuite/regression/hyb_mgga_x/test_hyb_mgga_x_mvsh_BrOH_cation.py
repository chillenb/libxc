
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_mvsh_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.736955956140642e+01, -1.736961214825386e+01, -1.736989949667002e+01, -1.736911851869926e+01, -1.736952008669684e+01, -1.736952008669684e+01, -2.521144343089767e+00, -2.521329591913196e+00, -2.526574999262481e+00, -2.527589714339703e+00, -2.526022915167655e+00, -2.526022915167655e+00, -4.347125127893054e-01, -4.341391838133613e-01, -4.207430287071673e-01, -4.271499065151865e-01, -4.270831940671910e-01, -4.270831940671910e-01, -1.337931906995397e-01, -1.354535404589010e-01, -4.902935644760886e-01, -7.861861900522571e-02, -1.217379331362443e-01, -1.217379331362442e-01, -8.228158033393414e-04, -8.935822647520037e-04, -1.088740591214669e-02, -3.540728656142255e-04, -6.036142285923127e-04, -6.036142285923129e-04, -4.252570409306789e+00, -4.253376459586452e+00, -4.252638239104086e+00, -4.253348532021946e+00, -4.252963180516272e+00, -4.252963180516272e+00, -1.549802149130625e+00, -1.567679893265297e+00, -1.545498822190842e+00, -1.561875383014959e+00, -1.561893315460092e+00, -1.561893315460092e+00, -4.603578418625629e-01, -5.012888125565140e-01, -4.104548764035003e-01, -4.250456709724569e-01, -4.698827461174202e-01, -4.698827461174202e-01, -4.566781756799916e-02, -1.269262228118991e-01, -4.099649768491460e-02, -1.464941227052342e+00, -6.214401066088499e-02, -6.214401066088500e-02, -2.485621986153920e-04, -3.491919089150966e-04, -2.947026476211548e-04, -2.321488606728703e-02, -3.792747500182739e-04, -3.792747500182740e-04, -4.790245467924557e-01, -4.763173812827316e-01, -4.772705410832957e-01, -4.780545803194143e-01, -4.776608237109913e-01, -4.776608237109913e-01, -4.635724095624584e-01, -3.986906468599714e-01, -4.171822066419899e-01, -4.355329028180403e-01, -4.259290318729846e-01, -4.259290318729846e-01, -5.211230389124404e-01, -1.694011644220895e-01, -2.032706009249968e-01, -2.649215889147221e-01, -2.338405827895707e-01, -2.338405827895707e-01, -3.560849481291154e-01, -1.010196710444382e-02, -1.612035603959712e-02, -2.590886440495795e-01, -3.469604068526979e-02, -3.469604068526980e-02, -1.334286959392532e-03, -5.897706344383465e-05, -1.640625373636627e-04, -3.207080976350058e-02, -3.415306311686715e-04, -3.415306311686711e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_mvsh_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.217332637789892e+01, -2.217301015402771e+01, -2.217343153193784e+01, -2.217308656563543e+01, -2.217369577044308e+01, -2.217348544332339e+01, -2.217250721506166e+01, -2.217197059965148e+01, -2.217339162474210e+01, -2.217252322528726e+01, -2.217339162474210e+01, -2.217252322528726e+01, -3.881688300929921e+00, -3.881197483257256e+00, -3.882071412787833e+00, -3.881435188647391e+00, -3.889058482252683e+00, -3.890286083072147e+00, -3.886971145681416e+00, -3.887053793003499e+00, -3.880218959414120e+00, -3.892752586317114e+00, -3.880218959414120e+00, -3.892752586317114e+00, -6.113873026249874e-01, -6.152486195171146e-01, -6.100839854719826e-01, -6.145983818728064e-01, -5.923515369002327e-01, -5.881687322884885e-01, -5.989024681228143e-01, -6.013748934723233e-01, -6.199982309661906e-01, -5.740019437155257e-01, -6.199982309661906e-01, -5.740019437155257e-01, -1.355372968912738e-01, -1.311676979058005e-01, -1.314231144613813e-01, -1.406200464752742e-01, -6.622220442045885e-01, -6.916272124856431e-01, -1.213279570472886e-01, -1.099918962606416e-01, -1.274210490741886e-01, -9.016437009997735e-02, -1.274210490741887e-01, -9.016437009997733e-02, -1.405491801539168e-03, -6.553050170369319e-04, -1.617479085458614e-03, -1.593532263876481e-03, -2.036081608545761e-02, -2.189841645787438e-02, -7.178548483828418e-04, -6.975187266589741e-04, -1.302449806502756e-03, -6.327631102003509e-04, -1.302449806502758e-03, -6.327631102003510e-04, -5.464258550017169e+00, -5.462809309271194e+00, -5.466953051490142e+00, -5.465413144641081e+00, -5.464377652041142e+00, -5.462876720551000e+00, -5.466691021905499e+00, -5.465243425823466e+00, -5.465662351833828e+00, -5.464121409393364e+00, -5.465662351833828e+00, -5.464121409393364e+00, -1.885865059848519e+00, -1.883378223237822e+00, -1.893263934187771e+00, -1.890562531859601e+00, -1.849267935192170e+00, -1.857778168607077e+00, -1.853530270967763e+00, -1.862142812771977e+00, -1.915388610716263e+00, -1.890735282219140e+00, -1.915388610716263e+00, -1.890735282219140e+00, -6.159724790952845e-01, -6.141753147925457e-01, -6.871569296718657e-01, -6.877260764486167e-01, -5.929966196843786e-01, -5.818934685434742e-01, -6.711970366087969e-01, -6.355776584394609e-01, -6.425742236835443e-01, -6.109189274166307e-01, -6.425742236835442e-01, -6.109189274166307e-01, -8.076115893349210e-02, -8.261322721356387e-02, -1.013692427413506e-01, -8.936766312197775e-02, -7.290795687520374e-02, -7.668119377462029e-02, -2.141985924095373e+00, -2.140653114565893e+00, -8.707055126492423e-02, -7.067502672403572e-02, -8.707055126492418e-02, -7.067502672403529e-02, -4.867034109611037e-04, -5.064077766411155e-04, -6.938774699908483e-04, -7.026468612868695e-04, -5.575919654085774e-04, -6.158548518414313e-04, -4.496689940936287e-02, -4.395227765971967e-02, -5.275043012353289e-04, -8.365959207018740e-04, -5.275043012353288e-04, -8.365959207018738e-04, -6.384502412993676e-01, -6.409963630599992e-01, -6.287168984317854e-01, -6.313255858201534e-01, -6.321334494603509e-01, -6.347408704928660e-01, -6.349831240770848e-01, -6.375315251221387e-01, -6.335592929967369e-01, -6.361358792940766e-01, -6.335592929967369e-01, -6.361358792940766e-01, -6.236375862370491e-01, -6.256824053823837e-01, -4.809029335923709e-01, -4.836105991305971e-01, -5.238683646388607e-01, -5.267806849403064e-01, -5.652922116446850e-01, -5.674466416570730e-01, -5.449906830778286e-01, -5.470944691159680e-01, -5.449906830778286e-01, -5.470944691159680e-01, -7.192221410338475e-01, -7.205916213336976e-01, -1.638983392060167e-01, -1.662475824584233e-01, -2.323708268043849e-01, -2.359824688383992e-01, -3.462439680045658e-01, -3.483678973364457e-01, -2.894292063920048e-01, -2.896342633654101e-01, -2.894292063920048e-01, -2.896342633654101e-01, -4.561474172473218e-01, -4.590669188189160e-01, -1.756192523070863e-02, -1.856564838302877e-02, -3.052441770210851e-02, -3.206389881791227e-02, -3.433040634723235e-01, -3.471290383451328e-01, -5.936663765931975e-02, -6.376017689039808e-02, -5.936663765931977e-02, -6.376017689039808e-02, -2.589504720870751e-03, -2.738850751715754e-03, -1.135244870264088e-04, -1.223536599490328e-04, -3.100808478042381e-04, -3.427820183025377e-04, -5.505543561948212e-02, -3.224111909231791e-02, -5.323438326468332e-04, -7.392864961968690e-04, -5.323438326468326e-04, -7.392864961968683e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mvsh_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.524616948615876e-08, 0.000000000000000e+00, -1.524918801487595e-08, -1.524602181710906e-08, 0.000000000000000e+00, -1.524908031238493e-08, -1.524586404779102e-08, 0.000000000000000e+00, -1.524876560714333e-08, -1.524755261726735e-08, 0.000000000000000e+00, -1.525091350428034e-08, -1.524607678656978e-08, 0.000000000000000e+00, -1.525028427896662e-08, -1.524607678656978e-08, 0.000000000000000e+00, -1.525028427896662e-08, -3.897100997412919e-05, 0.000000000000000e+00, -3.887389052633925e-05, -3.899968946789812e-05, 0.000000000000000e+00, -3.888564565529449e-05, -3.945163708398124e-05, 0.000000000000000e+00, -3.950764567099165e-05, -3.959081751807786e-05, 0.000000000000000e+00, -3.952322317315008e-05, -3.893941703886252e-05, 0.000000000000000e+00, -3.988565880341885e-05, -3.893941703886252e-05, 0.000000000000000e+00, -3.988565880341885e-05, -2.563621932016492e-03, 0.000000000000000e+00, -2.623174181608470e-03, -2.558747066600608e-03, 0.000000000000000e+00, -2.610231070715874e-03, -2.390091776018896e-03, 0.000000000000000e+00, -2.528573908046935e-03, -3.011166897042498e-03, 0.000000000000000e+00, -3.115998228307819e-03, -2.479368845753330e-03, 0.000000000000000e+00, -3.968347317187180e-03, -2.479368845753330e-03, 0.000000000000000e+00, -3.968347317187180e-03, -2.258967704581662e+00, 0.000000000000000e+00, -2.496268891472643e+00, -2.609349660781332e+00, 0.000000000000000e+00, -2.219842404932785e+00, -1.693434602755804e-04, 0.000000000000000e+00, -2.809731572509348e-04, -2.251421300197446e-02, 0.000000000000000e+00, -1.514148378548651e+00, -2.575729862457990e+00, 0.000000000000000e+00, 1.509443240309447e+00, -2.575729862457980e+00, 0.000000000000000e+00, 1.509443240309448e+00, 3.105717011952684e+01, 0.000000000000000e+00, -6.163945332572292e+02, 1.092056448286443e+02, 0.000000000000000e+00, -3.045618398496731e+01, 7.262837810352199e+00, 0.000000000000000e+00, 5.842411241832279e+00, 3.425689322059457e+02, 0.000000000000000e+00, 3.450182837803868e+02, 1.868409735494818e+02, 0.000000000000000e+00, 8.350721263317851e+02, 1.868409735494841e+02, 0.000000000000000e+00, 8.350721263317860e+02, -4.539291374367600e-06, 0.000000000000000e+00, -4.544612490385485e-06, -4.536300104713029e-06, 0.000000000000000e+00, -4.541717855936681e-06, -4.539496195941816e-06, 0.000000000000000e+00, -4.544780880085462e-06, -4.536914140986917e-06, 0.000000000000000e+00, -4.542136373797300e-06, -4.537478822889553e-06, 0.000000000000000e+00, -4.543111746699612e-06, -4.537478822889553e-06, 0.000000000000000e+00, -4.543111746699612e-06, -1.593067390581341e-04, 0.000000000000000e+00, -1.609255895025027e-04, -1.626215956387421e-04, 0.000000000000000e+00, -1.641049441968143e-04, -1.694560486668417e-04, 0.000000000000000e+00, -1.679430198847028e-04, -1.731918442634587e-04, 0.000000000000000e+00, -1.717128044214209e-04, -1.545225626897189e-04, 0.000000000000000e+00, -1.617188928208607e-04, -1.545225626897189e-04, 0.000000000000000e+00, -1.617188928208607e-04, -1.764821861249518e-02, 0.000000000000000e+00, -1.811231424134417e-02, -1.313047692582589e-02, 0.000000000000000e+00, -1.330735186022769e-02, -7.323433064091281e-02, 0.000000000000000e+00, -2.517061521862471e-02, -6.247911442093232e-02, 0.000000000000000e+00, -2.877754081938415e-02, -1.647170969032252e-02, 0.000000000000000e+00, -1.829030627369032e-02, -1.647170969032253e-02, 0.000000000000000e+00, -1.829030627369032e-02, 4.186281722247153e-01, 0.000000000000000e+00, 5.920818134594037e-01, -2.388116621302114e+00, 0.000000000000000e+00, -2.713298142970709e+00, 1.254629487222799e+00, 0.000000000000000e+00, 6.242271250433435e-01, -2.287770241870334e-04, 0.000000000000000e+00, -2.287096013206518e-04, -1.586789603763240e+00, 0.000000000000000e+00, -5.924830891526967e+00, -1.586789603763252e+00, 0.000000000000000e+00, -5.924830891527039e+00, 6.530945952461102e+02, 0.000000000000000e+00, 5.632455036696522e+02, 4.093961500575749e+02, 0.000000000000000e+00, 3.808320142737858e+02, 1.873824051189561e+03, 0.000000000000000e+00, 1.837670718170028e+03, 3.915663394346470e+00, 0.000000000000000e+00, 2.319768380712498e+00, 1.176368639346654e+03, 0.000000000000000e+00, 6.439638768904695e+02, 1.176368639346652e+03, 0.000000000000000e+00, 6.439638768904622e+02, -2.628686792820966e-02, 0.000000000000000e+00, -2.593016886844096e-02, -2.689607895070450e-02, 0.000000000000000e+00, -2.652537274558953e-02, -2.667980343479721e-02, 0.000000000000000e+00, -2.631352905208927e-02, -2.650253803641893e-02, 0.000000000000000e+00, -2.614108911795750e-02, -2.659000817613786e-02, 0.000000000000000e+00, -2.622656862722184e-02, -2.659000817613786e-02, 0.000000000000000e+00, -2.622656862722184e-02, -2.757065445729265e-02, 0.000000000000000e+00, -2.732506210322658e-02, -5.183543019240940e-02, 0.000000000000000e+00, -5.103778234702499e-02, -4.268777024987657e-02, 0.000000000000000e+00, -4.203209958141146e-02, -3.557891777324336e-02, 0.000000000000000e+00, -3.511514495633750e-02, -3.868001211810275e-02, 0.000000000000000e+00, -3.828070818840149e-02, -3.868001211810274e-02, 0.000000000000000e+00, -3.828070818840147e-02, -1.016761407539607e-02, 0.000000000000000e+00, -1.025044887589549e-02, -9.173147597338397e-01, 0.000000000000000e+00, -8.915685723415572e-01, -4.005541914589693e-01, 0.000000000000000e+00, -3.880959314456786e-01, -2.051812046056156e-01, 0.000000000000000e+00, -2.110780979137515e-01, -2.304332872645449e-01, 0.000000000000000e+00, -2.308507954364251e-01, -2.304332872645450e-01, 0.000000000000000e+00, -2.308507954364251e-01, -5.034219377950674e-02, 0.000000000000000e+00, -5.084402223632640e-02, -9.373041163848339e-01, 0.000000000000000e+00, 2.174575916492813e+00, 4.320842296166721e+00, 0.000000000000000e+00, 3.960068009901913e+00, -2.267506392161872e-01, 0.000000000000000e+00, -1.562593095142090e-01, 1.046046713089690e+00, 0.000000000000000e+00, 6.700038479990601e-02, 1.046046713089706e+00, 0.000000000000000e+00, 6.700038479989696e-02, 8.545300738810522e+01, 0.000000000000000e+00, 8.209426441991877e+01, 6.502792483045559e+03, 0.000000000000000e+00, 9.264344090895262e+03, 1.649400481230873e+03, 0.000000000000000e+00, 1.563194151233175e+03, 1.325688606475328e-01, 0.000000000000000e+00, -1.489939592914370e+01, 2.005888247365131e+03, 0.000000000000000e+00, 7.683066591046810e+02, 2.005888247365135e+03, 0.000000000000000e+00, 7.683066591046872e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mvsh_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mvsh_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mvsh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.187171331357344e-03, 1.187400333737375e-03, 1.187171035987794e-03, 1.187400088036691e-03, 1.187197063921744e-03, 1.187429801304338e-03, 1.187201944068630e-03, 1.187435673376557e-03, 1.187170664642482e-03, 1.187453007826554e-03, 1.187170664642482e-03, 1.187453007826554e-03, 1.306768145210696e-02, 1.303463833421437e-02, 1.307737107259665e-02, 1.303859508280255e-02, 1.323023636844194e-02, 1.324968249708388e-02, 1.327976514582704e-02, 1.325764897187816e-02, 1.305064895162369e-02, 1.338126999791699e-02, 1.305064895162369e-02, 1.338126999791699e-02, 6.592802690114401e-03, 6.782383537471556e-03, 6.562305514418417e-03, 6.741978369868178e-03, 5.950924027996811e-03, 6.199646950208422e-03, 7.434662536235471e-03, 7.698831241974936e-03, 6.534195044488623e-03, 9.035005829379071e-03, 6.534195044488623e-03, 9.035005829379071e-03, 9.605019149350734e-02, 1.164463537694196e-01, 1.129865236440980e-01, 1.090544451674906e-01, 1.010947101130781e-03, 1.432831995341997e-03, 1.170253074573149e-02, 3.938487590383852e-02, 1.216604151209545e-01, 1.489344482237088e-03, 1.216604151209540e-01, 1.489344482237089e-03, 2.191684078917524e-04, 1.462670232588521e-03, 9.846416254307008e-05, 3.971930245973554e-04, 5.351583898249906e-04, 9.227248399395990e-04, 2.887938040401340e-07, 3.742746234956599e-07, 1.557587009903088e-05, 1.264151210711855e-07, 1.557587009902931e-05, 1.264151210712071e-07, 5.110577581488310e-03, 5.112895735471523e-03, 5.109871319551615e-03, 5.112211680301392e-03, 5.110970370836873e-03, 5.113183819847929e-03, 5.110347157309529e-03, 5.112545746205442e-03, 5.109889250830028e-03, 5.112497782391966e-03, 5.109889250830028e-03, 5.112497782391966e-03, 9.764031606177021e-03, 9.859159122610088e-03, 1.016929496083464e-02, 1.025177279904054e-02, 1.020232179538848e-02, 1.015912264134465e-02, 1.060945314629979e-02, 1.057112484017154e-02, 9.704758666067894e-03, 1.002755826875103e-02, 9.704758666067894e-03, 1.002755826875103e-02, 2.873975049955323e-02, 2.925617475423251e-02, 2.705044774216249e-02, 2.744849427774197e-02, 8.838591087632700e-02, 3.407765713456369e-02, 8.487620189690770e-02, 4.318532167507412e-02, 3.010660544663719e-02, 2.871719090535398e-02, 3.010660544663719e-02, 2.871719090535398e-02, 7.731571778070157e-03, 6.783640211320930e-03, 1.080741317039871e-01, 1.230063620191943e-01, 4.022733518522734e-03, 6.899118805651191e-03, 1.287702367098716e-02, 1.285589955609603e-02, 2.650085678157368e-02, 7.642259806445810e-02, 2.650085678157379e-02, 7.642259806445888e-02, 8.220416609794808e-09, 9.508686391659345e-09, 7.050789927090921e-08, 5.535426540284634e-08, 7.478140653399158e-07, 9.246798954991710e-07, 1.273958995854225e-03, 3.461451844293196e-03, 9.549888855178154e-09, 2.815998306659096e-05, 9.549888855158973e-09, 2.815998306659119e-05, 4.253620770721658e-02, 4.244648242163417e-02, 4.283357242911170e-02, 4.273803050896647e-02, 4.272785344406210e-02, 4.263491453655294e-02, 4.264192149494257e-02, 4.254904635065497e-02, 4.268317599576790e-02, 4.259088417422037e-02, 4.268317599576790e-02, 4.259088417422036e-02, 4.110160455129368e-02, 4.114059706327176e-02, 4.958218562092467e-02, 4.944475125489060e-02, 4.662541787398285e-02, 4.651321491117266e-02, 4.415477567656616e-02, 4.403801032818892e-02, 4.501739325260923e-02, 4.504653779868105e-02, 4.501739325260922e-02, 4.504653779868103e-02, 2.410615772382194e-02, 2.441681120973933e-02, 8.265177327859038e-02, 8.170989317791540e-02, 6.108740555960206e-02, 6.084395289366531e-02, 6.907772618250257e-02, 7.216979856780707e-02, 5.284473179949487e-02, 5.286561615771877e-02, 5.284473179949487e-02, 5.286561615771877e-02, 3.769581332534060e-02, 3.881259336601237e-02, 2.872012535661684e-03, 1.962297574556132e-03, 9.700248654366590e-04, 1.217758920967281e-03, 6.714777942260756e-02, 4.862783060501710e-02, 6.840422227604290e-03, 1.134886463239131e-02, 6.840422227604257e-03, 1.134886463239134e-02, 1.005009170533817e-06, 1.003049723327595e-06, 1.468304270420772e-10, 1.099549409074087e-10, 4.052486513109792e-07, 4.976957845042231e-07, 9.494837467102236e-03, 4.822053056910907e-02, 3.020267957508181e-08, 2.058823164187305e-05, 3.020267957518035e-08, 2.058823164187305e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05