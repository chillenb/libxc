
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_blypr_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blypr", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.092054428437217e+01, -2.092056808680261e+01, -2.092075143006564e+01, -2.092036011725657e+01, -2.092055619202496e+01, -2.092055619202496e+01, -3.343370645768219e+00, -3.343345037360298e+00, -3.342785491475656e+00, -3.344491210543217e+00, -3.343372369351737e+00, -3.343372369351737e+00, -5.323191394804512e-01, -5.319638251211780e-01, -5.232511038241734e-01, -5.281495058405352e-01, -5.321879053306922e-01, -5.321879053306922e-01, -6.840467780384391e-02, -7.015075226581623e-02, -6.572965737570731e-01, -3.616417409838787e-02, -6.892709339407944e-02, -6.892709339407944e-02, -1.510994740632806e-05, -1.736407291691372e-05, -2.415457525634388e-03, -1.668714912380770e-06, -1.682694313344363e-05, -1.682694313344363e-05, -4.921282807749717e+00, -4.920716541555652e+00, -4.921231201117989e+00, -4.920790795664137e+00, -4.920983474270237e+00, -4.920983474270237e+00, -1.940042979790455e+00, -1.950765982048290e+00, -1.939981814159689e+00, -1.948338743365984e+00, -1.947381276419158e+00, -1.947381276419158e+00, -4.288677243997634e-01, -4.635650235504896e-01, -4.007558889099815e-01, -4.134725559857431e-01, -4.486739943670770e-01, -4.486739943670770e-01, -1.886305522612453e-02, -6.907420466870168e-02, -1.831711252727482e-02, -1.685634651928312e+00, -2.579690962053890e-02, -2.579690962053890e-02, -1.498107119102041e-06, -2.236084777664339e-06, -9.415040667152308e-07, -6.114241417081803e-03, -1.691487924167641e-06, -1.691487924167640e-06, -4.298911439816250e-01, -4.302013311664394e-01, -4.301273162554114e-01, -4.300409516350718e-01, -4.300867924736680e-01, -4.300867924736680e-01, -4.115071357785073e-01, -3.603804194762578e-01, -3.765895912096690e-01, -3.906255502200959e-01, -3.835175802009576e-01, -3.835175802009576e-01, -4.929852638739869e-01, -1.066855131537147e-01, -1.447632117786725e-01, -2.129145571345900e-01, -1.767954051520567e-01, -1.767954051520566e-01, -3.122224692949604e-01, -1.895134013703227e-03, -4.630625395224451e-03, -1.927434931728336e-01, -1.169607877311697e-02, -1.169607877311692e-02, -2.432441048047792e-05, -4.589108667326040e-08, -3.145836044748142e-07, -1.045803960892524e-02, -1.038314149684336e-06, -1.038314149684331e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_blypr_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blypr", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.512117210407749e+01, -2.512125404639554e+01, -2.512163467998902e+01, -2.512029255636045e+01, -2.512121519168463e+01, -2.512121519168463e+01, -3.974408904347372e+00, -3.974437485360855e+00, -3.975390681172019e+00, -3.974609524339686e+00, -3.974440902521104e+00, -3.974440902521104e+00, -6.817322424528945e-01, -6.809881330451378e-01, -6.631511209448636e-01, -6.688712588859057e-01, -6.814602010824720e-01, -6.814602010824720e-01, -1.117252069282515e-01, -1.142202907153101e-01, -8.368815118803707e-01, -6.079939453167428e-02, -1.124695845354798e-01, -1.124695845354798e-01, -3.021229065845301e-05, -3.471938589519223e-05, -4.888909123346576e-03, -3.337316484530283e-06, -3.364522824249029e-05, -3.364522824249029e-05, -6.104497344832032e+00, -6.106801442664347e+00, -6.104733646445019e+00, -6.106525183706568e+00, -6.105677498554372e+00, -6.105677498554372e+00, -2.148947448219328e+00, -2.165375608318336e+00, -2.141490682002420e+00, -2.154257634288373e+00, -2.169902714727406e+00, -2.169902714727406e+00, -5.797309443481794e-01, -6.462440245869566e-01, -5.441216962533721e-01, -5.760267257887880e-01, -6.068937766330057e-01, -6.068937766330057e-01, -3.242069871586512e-02, -1.166904323284511e-01, -3.151869677696656e-02, -2.215992566203086e+00, -4.368285292935520e-02, -4.368285292935520e-02, -2.996121048282403e-06, -4.471968654874963e-06, -1.882961507718061e-06, -1.128195600945982e-02, -3.382849259359445e-06, -3.382849259359444e-06, -6.107597412730729e-01, -6.041333167767080e-01, -6.064081423052861e-01, -6.082367722365991e-01, -6.073134720941613e-01, -6.073134720941613e-01, -5.874955004603342e-01, -4.876205704527346e-01, -5.136035990955927e-01, -5.397146605848546e-01, -5.261533601561444e-01, -5.261533601561444e-01, -6.832640975285971e-01, -1.732306011690105e-01, -2.224298160475258e-01, -3.068818467701468e-01, -2.616369337599623e-01, -2.616369337599622e-01, -4.292704679850683e-01, -3.941867958142179e-03, -8.758560090048951e-03, -2.818560711179547e-01, -2.063025437694806e-02, -2.063025437694799e-02, -4.865868797548171e-05, -9.178204926521417e-08, -6.291622161850622e-07, -1.856227688317167e-02, -2.076574955721476e-06, -2.076574955721466e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_blypr_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blypr", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.139584973570501e-09, -7.139555256378877e-09, -7.139315033753262e-09, -7.139803828598561e-09, -7.139570201568885e-09, -7.139570201568885e-09, -8.554036227371476e-06, -8.554327204269553e-06, -8.560963273564746e-06, -8.542511030090434e-06, -8.554044574213049e-06, -8.554044574213049e-06, -2.766522909278872e-03, -2.760975534088341e-03, -2.580610536703354e-03, -2.542072335437346e-03, -2.764570563967882e-03, -2.764570563967882e-03, -5.274773818264557e-03, -7.387580645489028e-03, -1.769899681156595e-03, 2.644081073174050e-03, -5.941711956938989e-03, -5.941711956938989e-03, -2.142765949453570e-04, -2.426099147048687e-04, -5.101965864959241e-03, -2.362007024417344e-05, -2.420051803440888e-04, -2.420051803440888e-04, -2.040035609553678e-06, -2.041527690615259e-06, -2.040176548396455e-06, -2.041336845411466e-06, -2.040816062832160e-06, -2.040816062832160e-06, -5.776561235421036e-05, -5.676803665595517e-05, -5.765919379494365e-05, -5.688273433319174e-05, -5.722541833801819e-05, -5.722541833801819e-05, -5.642839706936937e-03, -5.815330780839123e-03, -6.469357000489919e-03, -7.290823864451504e-03, -5.237018082052029e-03, -5.237018082052029e-03, -1.486760606458354e-02, 1.636726343006671e-02, -1.741893514079099e-02, -1.103684911219097e-04, -1.055663554752373e-02, -1.055663554752373e-02, -2.229328417565793e-05, -3.211589875921713e-05, -2.997052593953428e-05, -1.151702884137092e-02, -3.250888845531821e-05, -3.250888845523232e-05, -7.689121104656525e-03, -7.092520990346333e-03, -7.276630599503585e-03, -7.438580646734003e-03, -7.355136477268046e-03, -7.355136477268046e-03, -8.709961488595938e-03, -6.971032193429687e-03, -7.291514703943403e-03, -7.638449907719273e-03, -7.465540189313226e-03, -7.465540189313228e-03, -4.926198849972694e-03, 2.639615195097006e-03, -8.266611458078582e-03, -1.520024879406596e-02, -1.343157502243960e-02, -1.343157502243956e-02, -8.882398991021782e-03, -3.779000077905727e-03, -8.049743392482631e-03, -2.112878399164898e-02, -2.050352567677355e-02, -2.050352567677139e-02, -2.696743744615389e-04, -1.298972077086140e-06, -6.998324692162112e-06, -2.147245109540176e-02, -2.701960197388521e-05, -2.701960197366358e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05