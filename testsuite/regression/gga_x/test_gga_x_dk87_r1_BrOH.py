
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_dk87_r1_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.105799118270768e+01, -2.105801156919878e+01, -2.105818459652075e+01, -2.105784912096862e+01, -2.105800124775785e+01, -2.105800124775785e+01, -3.499102736648808e+00, -3.499070071209581e+00, -3.498316541997696e+00, -3.500356886969936e+00, -3.499100841635362e+00, -3.499100841635362e+00, -7.043549305595064e-01, -7.044031978349501e-01, -7.071717918285298e-01, -7.115295996216446e-01, -7.043688531842655e-01, -7.043688531842655e-01, -2.172210304541120e-01, -2.181899372624615e-01, -8.195678365907813e-01, -1.823692023118583e-01, -2.174870586266120e-01, -2.174870586266120e-01, -6.125770378008492e-02, -6.166850158941996e-02, -1.031704528466458e-01, -6.097561074629356e-02, -6.107466866276248e-02, -6.107466866276248e-02, -5.049449340309659e+00, -5.048602216526352e+00, -5.049369628466440e+00, -5.048710824914068e+00, -5.049005007526236e+00, -5.049005007526236e+00, -2.129801300982725e+00, -2.139675568032847e+00, -2.130942884619004e+00, -2.138638754141930e+00, -2.134971906379433e+00, -2.134971906379433e+00, -5.776268036127220e-01, -5.950960774575070e-01, -5.504909156597090e-01, -5.481237455759477e-01, -5.948290488168344e-01, -5.948290488168344e-01, -1.562449643282554e-01, -2.380235538641722e-01, -1.524352141339301e-01, -1.810345275049211e+00, -1.649532183928969e-01, -1.649532183928969e-01, -5.998663934224828e-02, -6.059210983069086e-02, -4.648656025403740e-02, -1.174393609445684e-01, -5.498319121244098e-02, -5.498319121244102e-02, -5.579279748710759e-01, -5.609663399382736e-01, -5.597700790015282e-01, -5.589083620570141e-01, -5.593296024749033e-01, -5.593296024749033e-01, -5.391980582895665e-01, -5.212712262465338e-01, -5.271162925672845e-01, -5.312222930668697e-01, -5.290559037644436e-01, -5.290559037644436e-01, -6.250850579847651e-01, -2.821496763950716e-01, -3.164496135668817e-01, -3.699991756199278e-01, -3.414312580767765e-01, -3.414312580767764e-01, -4.731880330524119e-01, -1.033104761424880e-01, -1.151805418083901e-01, -3.393662782509558e-01, -1.322401983164241e-01, -1.322401983164241e-01, -6.671623043490749e-02, -4.985288515579633e-02, -5.296038043344954e-02, -1.258612887048653e-01, -4.973994992520604e-02, -4.973994992520606e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_dk87_r1_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.490579995996017e+01, -2.490588108260077e+01, -2.490625762748462e+01, -2.490492896876970e+01, -2.490584261887525e+01, -2.490584261887525e+01, -4.016435126030024e+00, -4.016460527705753e+00, -4.017325835354369e+00, -4.016688330805271e+00, -4.016465299825189e+00, -4.016465299825189e+00, -7.626802134243821e-01, -7.618595750183416e-01, -7.420298249820342e-01, -7.477811365249261e-01, -7.623805833968743e-01, -7.623805833968743e-01, -2.116250061271414e-01, -2.140729799815695e-01, -9.175378362831914e-01, -1.577765515327435e-01, -2.123541267290454e-01, -2.123541267290454e-01, -1.292609326310645e-02, -1.353536878103758e-02, -5.415343885188970e-02, -6.222795475831176e-03, -1.339440086672939e-02, -1.339440086672939e-02, -6.133131621209825e+00, -6.135837125556542e+00, -6.133407657413908e+00, -6.135511266092994e+00, -6.134518164637792e+00, -6.134518164637792e+00, -2.237786188748060e+00, -2.253197292860248e+00, -2.231976596973559e+00, -2.243958734097834e+00, -2.255896208451897e+00, -2.255896208451897e+00, -6.733499097953192e-01, -7.626012996449441e-01, -6.376993398836780e-01, -6.867250285971974e-01, -7.022212666080132e-01, -7.022212666080132e-01, -1.183187088004418e-01, -2.167559172447008e-01, -1.166551373976394e-01, -2.319171549829705e+00, -1.353201412976811e-01, -1.353201412976811e-01, -6.003392875311890e-03, -6.858269841363809e-03, -5.140702795426041e-03, -7.476029111981060e-02, -6.248861637556349e-03, -6.248861637556376e-03, -7.368751095155917e-01, -7.225943241344142e-01, -7.276850467613207e-01, -7.316809082736004e-01, -7.296802480636829e-01, -7.296802480636829e-01, -7.147721583606550e-01, -5.763679409598180e-01, -6.072155812058141e-01, -6.420613191897157e-01, -6.234911936743908e-01, -6.234911936743909e-01, -7.979129280276391e-01, -2.694330591725960e-01, -3.164464350570990e-01, -4.013407516348391e-01, -3.554090925018187e-01, -3.554090925018187e-01, -5.194076649328156e-01, -5.028971319975247e-02, -6.756246403500214e-02, -3.807605741950870e-01, -9.643214123299110e-02, -9.643214123299110e-02, -1.514056723914736e-02, -1.881601059687251e-03, -3.571536332084331e-03, -9.197803154371141e-02, -5.311776839825656e-03, -5.311776839825597e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_dk87_r1_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.030346236034318e-09, -8.030319674344949e-09, -8.030069419061994e-09, -8.030506954220204e-09, -8.030333329740065e-09, -8.030333329740065e-09, -1.038761407625166e-05, -1.038810905462116e-05, -1.039999968176743e-05, -1.037071337476558e-05, -1.038769165351322e-05, -1.038769165351322e-05, -6.001808446566984e-03, -5.992844508245832e-03, -5.711222336239235e-03, -5.581801363843142e-03, -5.998696557262632e-03, -5.998696557262632e-03, -6.072455987499008e-01, -5.980496063558202e-01, -3.384599278306422e-03, -1.283650088179304e+00, -6.047196984291195e-01, -6.047196984291195e-01, -3.069571066670288e+03, -2.673703378497661e+03, -3.323513479281497e+01, -2.543507961884702e+04, -2.785448711349324e+03, -2.785448711349324e+03, -2.428394007436277e-06, -2.429560837246005e-06, -2.428502152332093e-06, -2.429409843243230e-06, -2.429012878895435e-06, -2.429012878895435e-06, -6.951568555188847e-05, -6.840231095594806e-05, -6.914028032717498e-05, -6.827460770732796e-05, -6.925643463359267e-05, -6.925643463359267e-05, -1.411066773245529e-02, -1.165612490873626e-02, -1.705100749532434e-02, -1.705368860484335e-02, -1.260733385029033e-02, -1.260733385029033e-02, -2.877902369641557e+00, -4.248342258042931e-01, -3.119920124005985e+00, -1.362462014308819e-04, -2.044360415047405e+00, -2.044360415047405e+00, -2.875248075447558e+04, -1.927663476222213e+04, -5.954764256450596e+04, -1.269740297986304e+01, -2.809353502410331e+04, -2.809353502410332e+04, -1.190364505421692e-02, -1.447808218938015e-02, -1.393232675927752e-02, -1.328424393674489e-02, -1.363941881693832e-02, -1.363941881693832e-02, -1.247673997117428e-02, -2.044211320147821e-02, -2.021655749089073e-02, -1.984772835863862e-02, -2.013568275167082e-02, -2.013568275167083e-02, -9.693272786451206e-03, -2.125688741797157e-01, -1.365334104738748e-01, -7.896860317201064e-02, -1.042439014940727e-01, -1.042439014940728e-01, -2.988764842198258e-02, -3.976512669533169e+01, -1.646751487814719e+01, -1.153474549606098e-01, -6.001058815389286e+00, -6.001058815389290e+00, -1.776391766836467e+03, -1.074910502679343e+06, -1.510770875791037e+05, -7.284275913103107e+00, -5.030668193359411e+04, -5.030668193359433e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05