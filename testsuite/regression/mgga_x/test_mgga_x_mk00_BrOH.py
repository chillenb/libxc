
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mk00_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.999625181563475e+01, -1.999645619312139e+01, -1.999747612700305e+01, -1.999414225089370e+01, -1.999635840657747e+01, -1.999635840657747e+01, -3.208975407226690e+00, -3.209461230530784e+00, -3.224123570008510e+00, -3.218517321448536e+00, -3.209118334307572e+00, -3.209118334307572e+00, -3.838792784485133e-01, -3.819781823751024e-01, -3.360182552861549e-01, -3.555376448900242e-01, -3.831918954067923e-01, -3.831918954067923e-01, -1.094629860743506e-01, -1.137986992002483e-01, -3.133830407781655e-01, -2.782943987779909e-02, -1.105398613286055e-01, -1.105398613286055e-01, -4.774254535053322e-05, -5.321231904980301e-05, -1.483720729319973e-03, -2.241178777557533e-06, -5.354974171580319e-05, -5.354974171580309e-05, 6.809909740024233e-01, 5.132138104096739e+00, -1.301768556036214e-01, 4.739409090150765e-01, -3.708419287425281e+00, 2.462273877329256e+01, -1.871435784274053e-02, 6.504913472613002e+00, -2.194442358553735e+00, -2.365120273717220e+00, -2.376002967379261e+00, -1.314455590213160e+00, 3.527845762478040e-02, 1.514171946377885e-01, 5.142062943652383e-02, -1.171047135056919e+00, -1.561329578613276e+00, -4.669011279838151e+01, -1.673009550666286e-03, 1.124553397254187e-02, -3.376714358237334e-04, -5.261397804752387e+00, -1.611141955945324e-02, -1.151860352561871e-02, 2.377308564371344e-07, 4.132253653510681e-10, -1.791057298262966e-10, -4.833600036846465e-04, 1.189091335817584e-06, -4.334159355603114e-08, 6.706007528092603e-06, -1.866189545736163e-05, -9.851964017239524e-07, 6.776576107563627e-04, -2.406528049709676e-05, 5.196132532826197e-07, -1.295936397473107e-06, -7.690835942896511e-01, 5.956020079052066e-02, -3.407805319894472e-01, 3.970685551093266e+00, 2.579658138628906e-02, 4.177427610704622e-02, -1.373818671391926e-03, -1.079070668314153e-03, -1.427250557562494e-02, 2.594759074793960e-01, -1.388608221553542e+00, -5.838796481983933e-01, 1.226053488324735e-04, -8.556055382858739e-03, -3.820884980218130e-01, -3.571017796226054e-04, -6.053521948781380e-03, -1.861301458611021e-06, 1.165139563375387e-09, -1.322668131477173e-08, -2.637726561141392e-03, -3.848425020822242e-07, -3.302135554707411e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mk00_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.998875544690426e+01, -5.998936857936417e+01, -5.999242838100914e+01, -5.998242675268111e+01, -5.998907521973240e+01, -5.998907521973240e+01, -9.626926221680071e+00, -9.628383691592351e+00, -9.672370710025529e+00, -9.655551964345607e+00, -9.627355002922716e+00, -9.627355002922716e+00, -1.151637835345540e+00, -1.145934547125307e+00, -1.008054765858465e+00, -1.066612934670073e+00, -1.149575686220377e+00, -1.149575686220377e+00, -3.283889582230516e-01, -3.413960976007447e-01, -9.401491223344969e-01, -8.348831963339727e-02, -3.316195839858166e-01, -3.316195839858165e-01, -1.432276360515997e-04, -1.596369571494090e-04, -4.451162187959920e-03, -6.723536332672599e-06, -1.606492251474096e-04, -1.606492251474093e-04, 2.042972922007270e+00, 1.539641431229022e+01, -3.905305668108641e-01, 1.421822727045229e+00, -1.112525786227584e+01, 7.386821631987770e+01, -5.614307352822157e-02, 1.951474041783901e+01, -6.583327075661205e+00, -7.095360821151657e+00, -7.128008902137783e+00, -3.943366770639480e+00, 1.058353728743412e-01, 4.542515839133655e-01, 1.542618883095715e-01, -3.513141405170758e+00, -4.683988735839827e+00, -1.400703383951453e+02, -5.019028651998859e-03, 3.373660191762560e-02, -1.013014307471200e-03, -1.578419341425716e+01, -4.833425867835969e-02, -3.455581057685610e-02, 7.131925693114030e-07, 1.239676096053204e-09, -5.373171894788899e-10, -1.450080011053940e-03, 3.567274007452751e-06, -1.300247806680934e-07, 2.011802258427780e-05, -5.598568637208489e-05, -2.955589205171856e-06, 2.032972832269088e-03, -7.219584149129027e-05, 1.558839759847859e-06, -3.887809192419323e-06, -2.307250782868953e+00, 1.786806023715620e-01, -1.022341595968342e+00, 1.191205665327980e+01, 7.738974415886717e-02, 1.253228283211387e-01, -4.121456014175777e-03, -3.237212004942460e-03, -4.281751672687481e-02, 7.784277224381878e-01, -4.165824664660629e+00, -1.751638944595180e+00, 3.678160464974206e-04, -2.566816614857621e-02, -1.146265494065439e+00, -1.071305338867816e-03, -1.816056584634414e-02, -5.583904375833065e-06, 3.495418690126162e-09, -3.968004394431520e-08, -7.913179683424176e-03, -1.154527506246672e-06, -9.906406664122232e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mk00_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mk00_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.097300403703526e-03, -1.097314843725162e-03, -1.097385249446670e-03, -1.097150319376260e-03, -1.097307937381721e-03, -1.097307937381721e-03, -6.548706744539433e-03, -6.550648535898053e-03, -6.608678177645264e-03, -6.584892799468679e-03, -6.549185208571557e-03, -6.549185208571557e-03, -1.330548278424180e-02, -1.321467566958147e-02, -1.108851957298896e-02, -1.212555438573200e-02, -1.327277730417183e-02, -1.327277730417183e-02, -5.284190724886742e-02, -5.488895871725565e-02, -5.150880564086881e-03, -8.962721991134428e-03, -5.324617009002295e-02, -5.324617009002293e-02, -5.769931808122068e-05, -6.237056717236403e-05, -7.221320141495577e-04, -1.151634602572566e-06, -6.518064886076886e-05, -6.518064886076867e-05, -8.797573533161371e-05, -4.993961205996661e-03, -3.214573054959010e-06, -4.259160361777587e-05, -2.608195171024284e-03, -1.149833981511495e-01, -1.253634070802534e-06, -1.483051890103479e-01, -1.738543292187890e-02, -1.986453136145266e-02, -1.970347183595383e-02, -6.030323844732827e-03, -1.702876063724991e-04, -2.447513052698776e-03, -4.234253504548054e-04, -1.935370459639881e-01, -2.978731780178421e-01, -2.663741461799218e+02, -8.167361261427329e-05, -5.458871362682244e-04, -3.458445041774302e-06, -1.050136137275285e-01, -4.898884306783470e-03, -2.503970617958703e-03, -1.443356258689692e-08, -2.921621240014883e-14, -1.303602459392867e-14, -2.829114030068399e-05, -3.198184687179397e-07, -4.248962131215473e-10, -5.600307330141201e-12, -4.407268768669406e-11, -1.221235008138445e-13, -5.752351232647354e-08, -7.270623374833874e-11, -3.389613897734798e-14, -2.306489418825970e-13, -1.244312225884311e-01, -6.548067780912780e-04, -1.908169011447493e-02, -2.747278088065393e+00, -1.159567257317460e-04, -1.616175229085033e-04, -4.094712747445086e-06, -1.507939511799748e-06, -1.262530591397633e-04, -6.031190701127407e-02, -1.727302677476740e+00, -9.774619677739028e-02, -6.206540932674457e-06, -1.218277014027001e-02, -1.072741757720987e-01, -6.959034026242814e-06, -1.999775179563757e-03, -5.447771854315975e-08, -1.131847151572361e-11, -2.127760124873672e-10, -4.372705903250847e-04, -5.457393326159273e-08, -4.017991290158687e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mk00_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mk00", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [8.778403229628211e-03, 8.778518749801295e-03, 8.779081995573359e-03, 8.777202555010079e-03, 8.778463499053767e-03, 8.778463499053767e-03, 5.238965395631547e-02, 5.240518828718442e-02, 5.286942542116211e-02, 5.267914239574943e-02, 5.239348166857245e-02, 5.239348166857245e-02, 1.064438622739344e-01, 1.057174053566518e-01, 8.870815658391169e-02, 9.700443508585599e-02, 1.061822184333747e-01, 1.061822184333747e-01, 4.227352579909394e-01, 4.391116697380452e-01, 4.120704451269504e-02, 7.170177592907542e-02, 4.259693607201836e-01, 4.259693607201834e-01, 4.615945446497655e-04, 4.989645373789123e-04, 5.777056113196462e-03, 9.213076820580524e-06, 5.214451908861509e-04, 5.214451908861494e-04, 7.038058826529097e-04, 3.995168964797328e-02, 2.571658443967208e-05, 3.407328289422070e-04, 2.086556136819427e-02, 9.198671852091960e-01, 1.002907256642027e-05, 1.186441512082784e+00, 1.390834633750312e-01, 1.589162508916213e-01, 1.576277746876306e-01, 4.824259075786261e-02, 1.362300850979993e-03, 1.958010442159021e-02, 3.387402803638443e-03, 1.548296367711905e+00, 2.382985424142737e+00, 2.130993169439374e+03, 6.533889009141863e-04, 4.367097090145795e-03, 2.766756033419442e-05, 8.401089098202282e-01, 3.919107445426776e-02, 2.003176494366962e-02, 1.154685006951754e-07, 2.337296992011906e-13, 1.042881967514294e-13, 2.263291224054719e-04, 2.558547749743518e-06, 3.399169704972379e-09, 4.480245864112960e-11, 3.525815014935525e-10, 9.769880065107561e-13, 4.601880986117883e-07, 5.816498699867099e-10, 2.711691118187838e-13, 1.845191535060776e-12, 9.954497807074485e-01, 5.238454224730224e-03, 1.526535209157994e-01, 2.197822470452315e+01, 9.276538058539677e-04, 1.292940183268026e-03, 3.275770197956069e-05, 1.206351609439798e-05, 1.010024473118107e-03, 4.824952560901926e-01, 1.381842141981392e+01, 7.819695742191223e-01, 4.965232746139566e-05, 9.746216112216008e-02, 8.581934061767893e-01, 5.567227220994251e-05, 1.599820143651005e-02, 4.358217483452780e-07, 9.054777212578885e-11, 1.702208099898937e-09, 3.498164722600678e-03, 4.365914660927418e-07, 3.214393032126950e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05