
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hapbe_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hapbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.695876403440844e+01, -1.695878089498171e+01, -1.695892130990120e+01, -1.695864391479542e+01, -1.695877238112462e+01, -1.695877238112462e+01, -2.851183688819066e+00, -2.851160275191507e+00, -2.850633153455533e+00, -2.852138356842765e+00, -2.851183628045818e+00, -2.851183628045818e+00, -5.906619919536100e-01, -5.905387696798692e-01, -5.892292667527191e-01, -5.929140372745548e-01, -5.906145550765367e-01, -5.906145550765367e-01, -1.846320532069103e-01, -1.858971182479087e-01, -6.892931850982459e-01, -1.489766310586367e-01, -1.849955267847270e-01, -1.849955267847270e-01, -1.361112837476691e-02, -1.425401047149930e-02, -5.661715001446067e-02, -6.540993046132723e-03, -1.410553285449373e-02, -1.410553285449373e-02, -4.112144579898042e+00, -4.111682631711409e+00, -4.112102327979304e+00, -4.111743039263763e+00, -4.111900262225294e+00, -4.111900262225294e+00, -1.743210886059842e+00, -1.751073121064088e+00, -1.744357749707609e+00, -1.750470387229261e+00, -1.747077805025242e+00, -1.747077805025242e+00, -5.008735168965507e-01, -5.336577702649651e-01, -4.775982906502434e-01, -4.904008682324031e-01, -5.173176439796137e-01, -5.173176439796137e-01, -1.180135493065643e-01, -1.976845177496497e-01, -1.160065265397691e-01, -1.524735660708686e+00, -1.312027344580675e-01, -1.312027344580675e-01, -6.310119586790305e-03, -7.210553933843716e-03, -5.404486532382138e-03, -7.705827513670410e-02, -6.569902044343034e-03, -6.569902044343034e-03, -5.086592485783742e-01, -5.063103567786605e-01, -5.070764249088197e-01, -5.077225755631758e-01, -5.073927207305423e-01, -5.073927207305423e-01, -4.936668114011287e-01, -4.455978120381626e-01, -4.575366065055636e-01, -4.697747295991314e-01, -4.633121917012666e-01, -4.633121917012666e-01, -5.577085420303500e-01, -2.366224565811196e-01, -2.674805668356929e-01, -3.198468368910764e-01, -2.913801786568365e-01, -2.913801786568365e-01, -4.054146463106697e-01, -5.272146735785543e-02, -7.017775484393171e-02, -2.999898126443326e-01, -9.716326828894342e-02, -9.716326828894342e-02, -1.594561804439485e-02, -1.975051211605406e-03, -3.751427265130599e-03, -9.264060735683287e-02, -5.583919910409771e-03, -5.583919910409760e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hapbe_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hapbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.009907759505900e+01, -2.009914925582166e+01, -2.009947114514040e+01, -2.009829764577939e+01, -2.009911536815003e+01, -2.009911536815003e+01, -3.313153593321283e+00, -3.313190201708149e+00, -3.314329267222718e+00, -3.313047827887658e+00, -3.313186095070913e+00, -3.313186095070913e+00, -6.753137992756219e-01, -6.740661973925802e-01, -6.417689104758191e-01, -6.471792125457124e-01, -6.748615131328738e-01, -6.748615131328738e-01, -1.988397631214691e-01, -2.020443370188197e-01, -8.156083084865180e-01, -1.557137451202510e-01, -1.997948299130111e-01, -1.997948299130111e-01, -1.807086841236689e-02, -1.891645571938493e-02, -7.187339906401395e-02, -8.714093057557486e-03, -1.871957989174478e-02, -1.871957989174478e-02, -5.057642016122910e+00, -5.060093407665335e+00, -5.057891802086872e+00, -5.059797896396205e+00, -5.058899991640488e+00, -5.058899991640488e+00, -1.813454454092663e+00, -1.827287850066559e+00, -1.806480726744787e+00, -1.817170008558088e+00, -1.832250994965982e+00, -1.832250994965982e+00, -6.256954522963203e-01, -6.950501844930809e-01, -5.951068548112843e-01, -6.351473822636654e-01, -6.510284201014013e-01, -6.510284201014013e-01, -1.329934544396985e-01, -2.017659487391044e-01, -1.300206969179849e-01, -1.977935471870378e+00, -1.411517253933597e-01, -1.411517253933597e-01, -8.406806050269288e-03, -9.604086007571873e-03, -7.198869219475026e-03, -9.369443713936797e-02, -8.750689717294226e-03, -8.750689717294232e-03, -6.650301224583665e-01, -6.601088143837365e-01, -6.620708672988598e-01, -6.634475765437283e-01, -6.627756553400688e-01, -6.627756553400688e-01, -6.452633353763874e-01, -5.335288111950477e-01, -5.688251794548411e-01, -6.006485490686387e-01, -5.847530564055493e-01, -5.847530564055494e-01, -7.256405480756785e-01, -2.429854496296049e-01, -2.870196790750399e-01, -3.833439010035590e-01, -3.313895468206314e-01, -3.313895468206313e-01, -4.843803451930621e-01, -6.761117473887272e-02, -8.722422399149021e-02, -3.735860522817237e-01, -1.118741777961197e-01, -1.118741777961197e-01, -2.115309401584644e-02, -2.633132269617611e-03, -5.000190806709745e-03, -1.066128181061830e-01, -7.438415348514377e-03, -7.438415348514364e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hapbe_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hapbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.319726062109807e-09, -6.319687510266718e-09, -6.319438351578621e-09, -6.320071175944119e-09, -6.319706376287320e-09, -6.319706376287320e-09, -7.611671310776790e-06, -7.611736087168001e-06, -7.612286577478160e-06, -7.605081791149924e-06, -7.611579285480560e-06, -7.611579285480560e-06, -3.587574060358205e-03, -3.603140375560796e-03, -3.947445125778110e-03, -3.840740129152555e-03, -3.593255479745279e-03, -3.593255479745279e-03, -3.507582628845473e-01, -3.372788901919631e-01, -1.826713185724718e-03, -6.375780863845548e-01, -3.468412861184619e-01, -3.468412861184619e-01, -3.450872491309108e+00, -3.459496817661096e+00, -1.439023919819838e+00, -2.449659394169849e+00, -3.593764454655589e+00, -3.593764454655589e+00, -1.647821985893931e-06, -1.646467953016430e-06, -1.647681396184280e-06, -1.646628810360663e-06, -1.647134805130693e-06, -1.647134805130693e-06, -5.834853260680561e-05, -5.722962146290695e-05, -5.832313954672725e-05, -5.745773180368406e-05, -5.759106791531637e-05, -5.759106791531637e-05, -4.655048219326059e-03, -1.184116321241985e-03, -5.647041148625898e-03, -2.455901110410820e-03, -3.873770940079629e-03, -3.873770940079629e-03, -7.750649601274925e-01, -2.568949352288893e-01, -8.836359250245237e-01, -5.057183316075798e-05, -8.096141468828073e-01, -8.096141468828073e-01, -2.598779096643172e+00, -2.603599986700166e+00, -7.451202276503595e+00, -1.400860320467248e+00, -3.845387514381659e+00, -3.845387514382325e+00, 4.650680520420273e-04, -1.063879213200695e-03, -5.770487620382450e-04, -1.568939369754698e-04, -3.721496448411607e-04, -3.721496448411620e-04, 9.235123875081409e-04, -9.130764537678806e-03, -6.716795336594076e-03, -4.344567496064210e-03, -5.566149831082570e-03, -5.566149831082570e-03, -1.203253480616004e-03, -1.408245895551646e-01, -8.609154766600155e-02, -3.148288230957338e-02, -5.474837637520887e-02, -5.474837637520891e-02, -1.313682478804014e-02, -1.219146800676919e+00, -1.212806836820893e+00, -3.032812523448658e-02, -1.321985409308087e+00, -1.321985409308090e+00, -2.592137438854620e+00, -4.482529991979579e+00, -3.858356107604881e+00, -1.616478536173827e+00, -5.616254670800724e+00, -5.616254670802125e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05