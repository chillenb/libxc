
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_p86vwn_ft_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn_ft", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.169112757626164e-01, -1.169105682250689e-01, -1.169088798949515e-01, -1.169193765582403e-01, -1.169136693178450e-01, -1.169136693178450e-01, -5.136325050632101e-02, -5.136619756627543e-02, -5.143501559165974e-02, -5.131854031883070e-02, -5.136713177982372e-02, -5.136713177982372e-02, -3.081773029820795e-02, -3.056284205370397e-02, -2.425707890155908e-02, -2.455566911546111e-02, -2.472383205372821e-02, -2.472383205372821e-02, -5.793168825018406e-03, -6.815744210644772e-03, -3.358388697271310e-02, 3.091580117614279e-03, -5.635029279531170e-04, -5.635029279531277e-04, -3.778665613262927e-03, -3.945471272358964e-03, -9.169616080548584e-03, -2.344744415224126e-03, -2.866652845049727e-03, -2.866652845049727e-03, -6.701300126595235e-02, -6.716071202159406e-02, -6.701908594369625e-02, -6.714946936352116e-02, -6.708803435728612e-02, -6.708803435728612e-02, -3.191032933658171e-02, -3.221564942382001e-02, -3.148175407623540e-02, -3.173206826146821e-02, -3.231694515419689e-02, -3.231694515419689e-02, -4.235613186985965e-02, -5.772957821955207e-02, -3.973960489841383e-02, -5.337459693179739e-02, -4.417063309965028e-02, -4.417063309965028e-02, 5.565462277543052e-03, 1.705881718670762e-03, 4.823393743897610e-03, -7.515638024718206e-02, 4.698301329464503e-03, 4.698301329464503e-03, -1.859410478369248e-03, -2.298007185517506e-03, -1.803656655402888e-03, -1.076512138429878e-03, -2.134804563500572e-03, -2.134804563500572e-03, -6.119615734839294e-02, -5.662199859531388e-02, -5.814354949938461e-02, -5.946878408387162e-02, -5.879803968960035e-02, -5.879803968960035e-02, -6.209277174588402e-02, -3.059713342428590e-02, -3.814834836313205e-02, -4.670763294378566e-02, -4.229268743013787e-02, -4.229268743013787e-02, -5.782064784277426e-02, -3.076774312754860e-03, -1.014659352262600e-02, -2.678517837567048e-02, -1.814778617382809e-02, -1.814778617382810e-02, -2.984841003316002e-02, -9.769241136098273e-03, -6.146580740621824e-03, -3.179184873805189e-02, 3.134310389089341e-03, 3.134310389089333e-03, -5.037425587025698e-03, -6.837004102529633e-04, -1.363451616325810e-03, 2.326958049889360e-03, -1.991153036378377e-03, -1.991153036378373e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_p86vwn_ft_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn_ft", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.834219639405298e-02, -5.834278071738084e-02, -5.834450985042081e-02, -5.833584012005195e-02, -5.834050382099093e-02, -5.834050382099093e-02, -9.424731664964754e-02, -9.425941105384360e-02, -9.453825630349068e-02, -9.404768365627202e-02, -9.426108513684209e-02, -9.426108513684209e-02, -8.620211048281182e-02, -8.602520824185024e-02, -8.042432174109779e-02, -8.077999729448880e-02, -8.094969764900861e-02, -8.094969764900861e-02, -4.657103261066409e-02, -4.794952183559866e-02, -8.996928084720461e-02, -2.529971700357633e-02, -3.607421682568900e-02, -3.607421682568898e-02, -4.819945787050642e-03, -5.012662096944248e-03, 6.193448527604306e-03, -3.037596037061936e-03, -3.694576487681382e-03, -3.694576487681382e-03, -1.189874323276650e-01, -1.193719817000117e-01, -1.190032947596379e-01, -1.193428120655717e-01, -1.191834322879388e-01, -1.191834322879388e-01, -5.651556125979570e-02, -5.786172104383915e-02, -5.296741521658785e-02, -5.416099913841532e-02, -5.898608813634527e-02, -5.898608813634527e-02, -8.580141675292496e-02, -8.217896394079162e-02, -8.382224639057098e-02, -8.004512812237123e-02, -8.598372017795204e-02, -8.598372017795204e-02, 5.276234693234939e-03, -3.406703522286379e-02, 8.509909530073696e-03, -1.177829922441940e-01, -1.336382640079908e-02, -1.336382640079908e-02, -2.417691867941946e-03, -2.977690487509357e-03, -2.304643134759697e-03, 1.723558506020792e-02, -2.754416030352903e-03, -2.754416030352903e-03, -7.561442730844639e-02, -7.935253551840429e-02, -7.822597354034336e-02, -7.715259103880293e-02, -7.770686631368250e-02, -7.770686631368250e-02, -7.329143727140328e-02, -8.060605660795712e-02, -8.247872996709385e-02, -8.180958327593318e-02, -8.249509138431532e-02, -8.249509138431532e-02, -8.414553930230063e-02, -4.634277735510390e-02, -5.781314602715583e-02, -7.184804135267538e-02, -6.609115962867397e-02, -6.609115962867397e-02, -7.860307975070181e-02, 3.832957427843900e-03, 1.575345262307389e-02, -7.081277814779861e-02, 7.463967691375173e-03, 7.463967691375147e-03, -6.298037495015690e-03, -8.986712656581964e-04, -1.779892293552320e-03, 1.046668534536448e-02, -2.567329446574701e-03, -2.567329446574694e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_p86vwn_ft_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_p86vwn_ft", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.992559865668116e-10, -1.992553885012146e-10, -1.992491082555300e-10, -1.992580659914428e-10, -1.992538936839399e-10, -1.992538936839399e-10, 6.093419262691037e-07, 6.096272840003890e-07, 6.161513195562572e-07, 6.041040570871558e-07, 6.095904518510985e-07, 6.095904518510985e-07, 1.948254402020512e-03, 1.938508165606005e-03, 1.672037741496696e-03, 1.627806719610879e-03, 1.655560570497071e-03, 1.655560570497071e-03, 3.169144177425228e-01, 3.164382064238941e-01, 1.029886654000805e-03, 4.299345322499074e-01, 4.159534237959005e-01, 4.159534237959004e-01, -5.430075793346300e+00, -7.262324174401164e+00, -1.024988751128136e+01, -5.160277908630367e-01, -3.299066391452922e+00, -3.299066391452949e+00, 1.948682323971529e-07, 1.973664743497860e-07, 1.949691669564291e-07, 1.971744199042586e-07, 1.961388372719504e-07, 1.961388372719504e-07, 2.073263084851823e-06, 2.181372900551394e-06, 1.618845337533563e-06, 1.719002463568205e-06, 2.359333920681281e-06, 2.359333920681281e-06, 5.563808307308524e-03, 6.264407127980385e-03, 7.343210064624591e-03, 9.623923116653528e-03, 5.477121085552331e-03, 5.477121085552331e-03, 1.306685829520725e-01, 1.739774407630614e-01, -6.552935830795958e-03, 5.469863635557069e-05, 5.666824039703138e-01, 5.666824039703138e-01, -4.682724582774090e-01, -9.167922362313278e-01, -2.314751541790472e+02, -2.580631304389660e+00, -3.423885243372552e+01, -3.423885243372536e+01, 1.062972520346984e-02, 9.015980905922103e-03, 9.466012445263189e-03, 9.916469772916360e-03, 9.680456852343510e-03, 9.680456852343510e-03, 1.295564296525738e-02, 8.445556391446141e-03, 8.862630517135098e-03, 9.582716016704530e-03, 9.192883072236611e-03, 9.192883072236611e-03, 5.006157334100563e-03, 9.514431847331754e-02, 6.427322105375481e-02, 3.699696235484647e-02, 4.987422645414924e-02, 4.987422645414926e-02, 1.210566108272243e-02, -1.017239133314703e+01, -5.778290318356223e+00, 5.239692669894651e-02, -6.160444783234392e-02, -6.160444783234530e-02, -6.742814926229830e+00, -1.029279529974542e+00, -2.381806918555751e+00, -4.956096808805970e-01, -5.905043257636718e+01, -5.905043257636739e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05