
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2b_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.244432012561023e+01, -2.244437261772703e+01, -2.244465400029173e+01, -2.244387273802031e+01, -2.244427525057189e+01, -2.244427525057189e+01, -3.405299485986428e+00, -3.405316251763518e+00, -3.406022911089365e+00, -3.407949448132813e+00, -3.406600170392571e+00, -3.406600170392571e+00, -6.653278337367943e-01, -6.650552644828231e-01, -6.596169232373481e-01, -6.646613313989549e-01, -6.659325552380673e-01, -6.659325552380673e-01, -2.000774883735744e-01, -2.020285805340517e-01, -7.681361548294504e-01, -1.598960012658727e-01, -1.934692430746106e-01, -1.934692430746105e-01, -8.426188611537778e-03, -8.872857113361807e-03, -4.829317862332487e-02, -4.859874476433356e-03, -6.782318243156402e-03, -6.782318243156402e-03, -5.448809659894705e+00, -5.449272115524566e+00, -5.448837911656007e+00, -5.449246031965201e+00, -5.449040260099075e+00, -5.449040260099075e+00, -2.117931006438373e+00, -2.134104618097289e+00, -2.114497588573493e+00, -2.128562883047984e+00, -2.128529884522248e+00, -2.128529884522248e+00, -5.990508873639877e-01, -6.297214223998018e-01, -5.387419125788430e-01, -5.362219394765086e-01, -6.094843456767134e-01, -6.094843456767135e-01, -1.185048997503041e-01, -2.075447616313342e-01, -1.105301069388953e-01, -1.817205728331911e+00, -1.346571306808925e-01, -1.346571306808925e-01, -3.752153670930858e-03, -4.750794751793510e-03, -3.637931447466840e-03, -7.644087988637727e-02, -4.569723213365099e-03, -4.569723213365100e-03, -6.093515169002519e-01, -6.082025595787902e-01, -6.086161863805148e-01, -6.089459674143511e-01, -6.087804328800565e-01, -6.087804328800565e-01, -5.894178733428277e-01, -5.273421172031019e-01, -5.453444994677287e-01, -5.628788202559005e-01, -5.537494919840349e-01, -5.537494919840349e-01, -6.504000959293479e-01, -2.546571046173449e-01, -2.956773966215692e-01, -3.609321610767754e-01, -3.285200248852285e-01, -3.285200248852285e-01, -4.784869509162231e-01, -4.623478815979985e-02, -6.244951521611904e-02, -3.442234551078144e-01, -9.507041434189439e-02, -9.507041434189442e-02, -1.187286442474403e-02, -1.269958579079563e-03, -2.670583220269492e-03, -8.981365721185749e-02, -4.196609933771152e-03, -4.196609933771149e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2b_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.864237017356371e+01, -2.864346245758996e+01, -2.864249713140218e+01, -2.864355451910058e+01, -2.864303647605638e+01, -2.864428991911549e+01, -2.864161038701272e+01, -2.864247061334054e+01, -2.864244348941855e+01, -2.864329505308472e+01, -2.864244348941855e+01, -2.864329505308472e+01, -4.190978324773095e+00, -4.191079333501940e+00, -4.191072369672123e+00, -4.191158805592629e+00, -4.193148059348552e+00, -4.193628598482328e+00, -4.191702409528534e+00, -4.191992091013598e+00, -4.190073905870250e+00, -4.193938256393439e+00, -4.190073905870250e+00, -4.193938256393439e+00, -7.862483815038102e-01, -7.905949145336594e-01, -7.846136894352801e-01, -7.899333716631092e-01, -7.634245306188804e-01, -7.562256837535445e-01, -7.642368896045455e-01, -7.664034475251631e-01, -7.985891934440082e-01, -7.224168758874560e-01, -7.985891934440082e-01, -7.224168758874560e-01, -2.137501912055190e-01, -2.287909393761053e-01, -2.134930003198428e-01, -2.307508031555854e-01, -8.973618399595494e-01, -9.361583770503596e-01, -1.687608941177181e-01, -1.743988695827431e-01, -2.205303520464040e-01, -1.379951807597017e-01, -2.205303520464039e-01, -1.379951807597016e-01, -1.085374558715567e-02, -1.153132520130802e-02, -1.137133316044271e-02, -1.217869060132687e-02, -6.103536711270160e-02, -6.418393076437162e-02, -6.530679501244689e-03, -6.422008350635374e-03, -9.685452478561268e-03, -5.514709537685626e-03, -9.685452478561268e-03, -5.514709537685626e-03, -7.081704838367819e+00, -7.080272272659903e+00, -7.083730082136960e+00, -7.082227450158581e+00, -7.081937383225870e+00, -7.080425990504771e+00, -7.083670258016081e+00, -7.082197515689133e+00, -7.082651569217990e+00, -7.081239240014487e+00, -7.082651569217990e+00, -7.081239240014487e+00, -2.423637364713333e+00, -2.428603262976643e+00, -2.466612669898554e+00, -2.470135887416144e+00, -2.429974223814480e+00, -2.432366121198881e+00, -2.468764384362400e+00, -2.471944243225418e+00, -2.445039926080276e+00, -2.450166272110577e+00, -2.445039926080276e+00, -2.450166272110577e+00, -7.342264954204430e-01, -7.337150072206076e-01, -8.601166123899673e-01, -8.624969236566241e-01, -6.425039302410266e-01, -6.755463681516471e-01, -6.944328021315144e-01, -7.326732643618319e-01, -7.776502107240810e-01, -7.290696894087684e-01, -7.776502107240812e-01, -7.290696894087686e-01, -1.384460573789651e-01, -1.389101309388407e-01, -2.280366750729882e-01, -2.285407305134967e-01, -1.279577149705685e-01, -1.331053723797767e-01, -2.466910807555077e+00, -2.466030896977699e+00, -1.498607287929248e-01, -1.529539042258191e-01, -1.498607287929248e-01, -1.529539042258192e-01, -4.900066694166936e-03, -5.092348308673297e-03, -6.283607577452758e-03, -6.378742192726452e-03, -4.697910657923626e-03, -4.974632945584243e-03, -9.466836505115729e-02, -9.547620100575002e-02, -4.799065883176429e-03, -6.587784781395812e-03, -4.799065883176429e-03, -6.587784781395815e-03, -8.080498511813642e-01, -8.112777390297086e-01, -7.967872851770761e-01, -8.001673754070471e-01, -8.005315272625865e-01, -8.038919071495341e-01, -8.038209063671359e-01, -8.070878400492698e-01, -8.021557771994491e-01, -8.054696203609055e-01, -8.021557771994491e-01, -8.054696203609055e-01, -7.921846936259553e-01, -7.945242984476847e-01, -6.435546434788405e-01, -6.467042370057077e-01, -6.765350618415936e-01, -6.800966096588049e-01, -7.151808710750234e-01, -7.179494066355822e-01, -6.937269860747427e-01, -6.970432003433746e-01, -6.937269860747426e-01, -6.970432003433746e-01, -8.875198839085993e-01, -8.941849444900788e-01, -2.827610568594860e-01, -2.839429480195503e-01, -3.214904612481594e-01, -3.244469684464254e-01, -4.026951313650243e-01, -4.048788338399981e-01, -3.585486194104145e-01, -3.583235583103900e-01, -3.585486194104146e-01, -3.583235583103901e-01, -5.464295911826726e-01, -5.546819581116363e-01, -6.007675811159131e-02, -6.045742401699532e-02, -7.869539865417946e-02, -8.095728571164421e-02, -3.942430720938275e-01, -4.090707373051690e-01, -1.118025530692288e-01, -1.147880551152494e-01, -1.118025530692288e-01, -1.147880551152494e-01, -1.550715910382776e-02, -1.606575493144211e-02, -1.691266485136893e-03, -1.695092046273321e-03, -3.440793968274479e-03, -3.659423058855445e-03, -1.077143440317740e-01, -1.095688724546368e-01, -4.545077377761179e-03, -6.039004955723975e-03, -4.545077377761175e-03, -6.039004955723968e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2b_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.335445975546454e-09, 0.000000000000000e+00, -9.319509513303719e-09, -9.335205384907109e-09, 0.000000000000000e+00, -9.319336689009932e-09, -9.332289427530874e-09, 0.000000000000000e+00, -9.315761318524399e-09, -9.334904373911069e-09, 0.000000000000000e+00, -9.319119871699945e-09, -9.335349578580801e-09, 0.000000000000000e+00, -9.316210734718469e-09, -9.335349578580801e-09, 0.000000000000000e+00, -9.316210734718469e-09, -2.555262101953108e-05, 0.000000000000000e+00, -2.549275314696444e-05, -2.556792142289001e-05, 0.000000000000000e+00, -2.549806483120816e-05, -2.581154567933958e-05, 0.000000000000000e+00, -2.584064110375546e-05, -2.593597223401656e-05, 0.000000000000000e+00, -2.588675671330046e-05, -2.553997250645108e-05, 0.000000000000000e+00, -2.610860751026128e-05, -2.553997250645108e-05, 0.000000000000000e+00, -2.610860751026128e-05, -7.056816360423994e-03, 0.000000000000000e+00, -7.041375003072853e-03, -7.070205720963983e-03, 0.000000000000000e+00, -7.047857239943695e-03, -7.272905964341065e-03, 0.000000000000000e+00, -7.465854808777174e-03, -7.607299515941620e-03, 0.000000000000000e+00, -7.632754583512374e-03, -6.978069149332800e-03, 0.000000000000000e+00, -1.006951128303582e-02, -6.978069149332800e-03, 0.000000000000000e+00, -1.006951128303582e-02, -1.014412642761725e+00, 0.000000000000000e+00, -7.045758597810036e-01, -1.099889041772651e+00, 0.000000000000000e+00, -7.525369206256134e-01, -4.706816466450349e-03, 0.000000000000000e+00, -4.393782987817173e-03, -1.508399851386463e+00, 0.000000000000000e+00, -1.382266876038799e+00, -9.409641416998198e-01, 0.000000000000000e+00, -1.766422795205135e+00, -9.409641416998188e-01, 0.000000000000000e+00, -1.766422795205137e+00, -3.421047298063017e+00, 0.000000000000000e+00, -3.386065567535256e+00, -3.604447342951080e+00, 0.000000000000000e+00, -3.585049806221304e+00, -2.001025923114547e+00, 0.000000000000000e+00, -2.003661074000841e+00, -3.136721783461407e+00, 0.000000000000000e+00, -3.051422540174771e+00, -3.416713996319283e+00, 0.000000000000000e+00, -8.676278777947511e+00, -3.416713996319293e+00, 0.000000000000000e+00, -8.676278777947527e+00, -1.558835962429639e-06, 0.000000000000000e+00, -1.553978560446508e-06, -1.557695783982589e-06, 0.000000000000000e+00, -1.552860272573347e-06, -1.556245790018274e-06, 0.000000000000000e+00, -1.552117580864103e-06, -1.555358370730609e-06, 0.000000000000000e+00, -1.551191612876728e-06, -1.560168177327768e-06, 0.000000000000000e+00, -1.553737083452549e-06, -1.560168177327768e-06, 0.000000000000000e+00, -1.553737083452549e-06, -1.576660752875591e-04, 0.000000000000000e+00, -1.548923439169657e-04, -1.427959087114849e-04, 0.000000000000000e+00, -1.405380158583049e-04, -1.410371007730059e-04, 0.000000000000000e+00, -1.434364159735209e-04, -1.275397016715060e-04, 0.000000000000000e+00, -1.294353865064449e-04, -1.609933634642340e-04, 0.000000000000000e+00, -1.488887138416752e-04, -1.609933634642340e-04, 0.000000000000000e+00, -1.488887138416752e-04, -5.172087912541877e-02, 0.000000000000000e+00, -5.190874903152589e-02, -5.259207062705762e-02, 0.000000000000000e+00, -5.308129846150687e-02, -5.260730130697831e-02, 0.000000000000000e+00, -5.938470191086411e-02, -3.529880893512852e-02, 0.000000000000000e+00, -4.558111645178748e-02, -4.409546948134516e-02, 0.000000000000000e+00, -5.671979492970712e-02, -4.409546948134514e-02, 0.000000000000000e+00, -5.671979492970711e-02, -1.451929918461952e+00, 0.000000000000000e+00, -1.495081280810854e+00, -5.340093749792542e-01, 0.000000000000000e+00, -5.377074905861032e-01, -1.651379370126691e+00, 0.000000000000000e+00, -1.585989781993242e+00, -3.460180321381595e-04, 0.000000000000000e+00, -3.469639034457617e-04, -1.565903300775600e+00, 0.000000000000000e+00, -1.822344929572814e+00, -1.565903300775600e+00, 0.000000000000000e+00, -1.822344929572812e+00, -4.403377428006416e+00, 0.000000000000000e+00, -3.811369561718490e+00, -3.792345715944355e+00, 0.000000000000000e+00, -3.501387463969495e+00, -2.159809436635699e+01, 0.000000000000000e+00, -2.402722546897276e+01, -2.383838070670089e+00, 0.000000000000000e+00, -2.230505007376052e+00, -1.073904939600508e+01, 0.000000000000000e+00, -1.054361621634947e+01, -1.073904939600505e+01, 0.000000000000000e+00, -1.054361621634946e+01, -2.949425045161960e-02, 0.000000000000000e+00, -2.859440515500385e-02, -2.620575410338471e-02, 0.000000000000000e+00, -2.547667451545449e-02, -2.727935845639547e-02, 0.000000000000000e+00, -2.649515298906061e-02, -2.823575341746121e-02, 0.000000000000000e+00, -2.740651705523656e-02, -2.776083952419891e-02, 0.000000000000000e+00, -2.695027948575325e-02, -2.776083952419891e-02, 0.000000000000000e+00, -2.695027948575326e-02, -5.462693211919580e-02, 0.000000000000000e+00, -5.263660242189690e-02, -2.791978599108352e-02, 0.000000000000000e+00, -2.748515602782322e-02, -3.312896731250268e-02, 0.000000000000000e+00, -3.256404129544756e-02, -4.042834045117634e-02, 0.000000000000000e+00, -3.983236816407527e-02, -3.810943111960288e-02, 0.000000000000000e+00, -3.685876175787620e-02, -3.810943111960290e-02, 0.000000000000000e+00, -3.685876175787626e-02, -3.863038160757142e-02, 0.000000000000000e+00, -4.033221204454024e-02, -2.881182756980000e-01, 0.000000000000000e+00, -2.878352825925898e-01, -2.826885254262381e-01, 0.000000000000000e+00, -2.807936368503048e-01, -2.782600661747271e-01, 0.000000000000000e+00, -2.737471558174800e-01, -2.915239522608242e-01, 0.000000000000000e+00, -2.946477573118063e-01, -2.915239522608241e-01, 0.000000000000000e+00, -2.946477573118064e-01, -9.439543262915312e-02, 0.000000000000000e+00, -8.948228885998516e-02, -1.802362917640630e+00, 0.000000000000000e+00, -1.808877464444735e+00, -1.812132821531413e+00, 0.000000000000000e+00, -1.848935669443545e+00, -3.893863005259628e-01, 0.000000000000000e+00, -3.781138376803584e-01, -2.338248972430617e+00, 0.000000000000000e+00, -2.739356282017066e+00, -2.338248972430619e+00, 0.000000000000000e+00, -2.739356282017074e+00, -2.716511298103823e+00, 0.000000000000000e+00, -2.773381446735150e+00, -1.354807764133014e+01, 0.000000000000000e+00, -2.399837338759771e+01, -8.361071974328100e+00, 0.000000000000000e+00, -8.900836326936002e+00, -2.560461792777899e+00, 0.000000000000000e+00, -2.341777195659571e+00, -2.214183801459491e+01, 0.000000000000000e+00, -1.094654417974930e+01, -2.214183801459495e+01, 0.000000000000000e+00, -1.094654417974933e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2b_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2b_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.284579291681885e-04, 3.272841214340815e-04, 3.284469336175164e-04, 3.272762676022277e-04, 3.282571487681699e-04, 3.270476544753663e-04, 3.283737622289108e-04, 3.271983978699660e-04, 3.284546767313763e-04, 3.270252998182724e-04, 3.284546767313763e-04, 3.270252998182724e-04, 3.376858139700625e-03, 3.359632797734877e-03, 3.380853850857660e-03, 3.360867650527048e-03, 3.442957604508803e-03, 3.450157841748350e-03, 3.484532585927138e-03, 3.470544529018588e-03, 3.369964007239212e-03, 3.527995924744010e-03, 3.369964007239212e-03, 3.527995924744010e-03, 1.961790085592915e-05, 9.336684820303491e-06, 2.660372005920743e-05, 9.383461296034194e-06, 1.595992574436512e-04, 4.020081530558216e-04, 8.649861731031251e-04, 8.978796973116931e-04, 8.583592878217216e-06, 4.501011224237043e-03, 8.583592878217216e-06, 4.501011224237043e-03, 1.493759799941741e-02, 7.068269455114848e-03, 1.828318785196433e-02, 9.943696063475855e-03, 5.939083017791477e-04, 7.088429606616256e-04, 6.888360843248818e-03, 7.616792451644981e-03, 1.695112328940633e-02, 8.278460530798604e-04, 1.695112328940630e-02, 8.278460530798576e-04, 6.489867870229220e-10, 3.741073797138411e-10, 1.453824341331287e-09, 1.160243788469617e-09, 1.458561529153210e-05, 1.753035943584014e-05, 3.348112664931838e-10, 2.681320219513085e-10, 1.059163305110936e-09, 7.291932584418816e-10, 1.059163305110969e-09, 7.291932584418814e-10, 1.229346337534122e-04, 1.159186398065184e-04, 1.219654846370071e-04, 1.149600906964062e-04, 1.200715023772793e-04, 1.138604015595490e-04, 1.193398518756376e-04, 1.130888118014172e-04, 1.245687501442933e-04, 1.157929311968050e-04, 1.245687501442933e-04, 1.157929311968050e-04, 5.394842885460042e-03, 5.264850498215564e-03, 4.852790221143075e-03, 4.734682498988955e-03, 4.485410222407618e-03, 4.632241335215423e-03, 3.957092642839194e-03, 4.082831023843179e-03, 5.759331274200114e-03, 5.080445849761702e-03, 5.759331274200114e-03, 5.080445849761702e-03, 5.426339873958302e-02, 5.455203858255880e-02, 6.987177236996182e-02, 7.161058970221221e-02, 2.840907258369000e-02, 4.415074357620267e-02, 1.391669223120793e-02, 2.951103368557348e-02, 5.448060468510307e-02, 5.682699480601500e-02, 5.448060468510304e-02, 5.682699480601498e-02, 1.329052905060968e-03, 1.458019535363279e-03, 5.730037986543000e-03, 6.079956326834857e-03, 9.807391806286119e-04, 1.198281459206666e-03, 7.904719289222914e-03, 7.921200946199792e-03, 2.767873987339496e-03, 4.359182191284503e-03, 2.767873987339492e-03, 4.359182191284492e-03, 3.652271734513867e-11, 4.920161144634729e-11, 3.951256613894158e-10, 3.546870888922491e-10, 1.185566750589265e-09, 1.899259971266916e-09, 2.669588417286764e-04, 2.351598459600785e-04, 7.019158078251556e-12, 6.120125998424560e-10, 7.019158078251913e-12, 6.120125998424473e-10, 3.102853782753678e-02, 3.023483345637065e-02, 2.554793398430544e-02, 2.496223368554033e-02, 2.733288667369372e-02, 2.668183306868280e-02, 2.892660373359439e-02, 2.821943551514796e-02, 2.813259906944246e-02, 2.744790690972922e-02, 2.813259906944246e-02, 2.744790690972924e-02, 6.019438368562774e-02, 5.840546277444329e-02, 1.110342419686428e-02, 1.105928885006336e-02, 1.961671376962767e-02, 1.952086760240771e-02, 3.224625448972973e-02, 3.209538077792343e-02, 2.694719735013441e-02, 2.612720858308315e-02, 2.694719735013444e-02, 2.612720858308321e-02, 5.321518593632831e-02, 5.781362623352539e-02, 6.327170192225041e-03, 6.616254415598244e-03, 1.892999313615994e-02, 1.963434185743085e-02, 5.214411731141743e-02, 5.186422276451048e-02, 3.649224010555337e-02, 3.691780785207142e-02, 3.649224010555333e-02, 3.691780785207143e-02, 4.397189696318338e-02, 4.288385339021037e-02, 5.891571691933693e-06, 7.264024055371037e-06, 5.367521480237064e-05, 6.427533900049908e-05, 6.355090231918521e-02, 7.360328800354667e-02, 7.419208584903886e-04, 1.218741945988429e-03, 7.419208584903899e-04, 1.218741945988442e-03, 2.127825109120252e-08, 2.475775774041001e-08, 3.943282405105273e-14, 2.218956082991283e-13, 3.559928216780717e-11, 5.498550228121804e-11, 6.476546433804706e-04, 3.870208023809883e-04, 1.535925054997187e-10, 4.287570315653887e-10, 1.535925054997179e-10, 4.287570315653995e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05