
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hjs_b88_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_b88", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.555255086388164e+01, -1.555257486825282e+01, -1.555272928680879e+01, -1.555233526395797e+01, -1.555256312734127e+01, -1.555256312734127e+01, -2.611592273785198e+00, -2.611582513924338e+00, -2.611419910073646e+00, -2.612242324982262e+00, -2.611598140547281e+00, -2.611598140547281e+00, -5.481143225135213e-01, -5.477918041161940e-01, -5.407669996733928e-01, -5.443310540743983e-01, -5.479954723156716e-01, -5.479954723156716e-01, -1.775253156447329e-01, -1.789530546623255e-01, -6.428958904514173e-01, -1.467549841750001e-01, -1.779404451475221e-01, -1.779404451475221e-01, -5.100225680179892e-02, -5.164424361528719e-02, -8.844177434091531e-02, -7.882440956957355e-02, -5.118928707301301e-02, -5.118928707301301e-02, -3.809727419874994e+00, -3.809792642003916e+00, -3.809737568256345e+00, -3.809788214678570e+00, -3.809755690577024e+00, -3.809755690577024e+00, -1.558252732867909e+00, -1.566457430897093e+00, -1.557492942640264e+00, -1.563865972073224e+00, -1.564866951847131e+00, -1.564866951847131e+00, -4.775467412022352e-01, -5.161643262709792e-01, -4.555964847653751e-01, -4.741478643068702e-01, -4.939387017548819e-01, -4.939387017548819e-01, -1.275619740380709e-01, -1.878179226755642e-01, -1.247485378495107e-01, -1.440100923218629e+00, -1.337933489309869e-01, -1.337933489309869e-01, -8.040468297349104e-02, -7.026599614454429e-02, -3.842130885744162e-09, -9.938183009454035e-02, -6.435247599732384e-02, -6.435247599732384e-02, -4.949336936746667e-01, -4.909632165347042e-01, -4.923175404209478e-01, -4.934143399689340e-01, -4.928598135154885e-01, -4.928598135154885e-01, -4.811160933415735e-01, -4.212656435914152e-01, -4.366925065101726e-01, -4.522718074580903e-01, -4.441609621300479e-01, -4.441609621300477e-01, -5.383289603583790e-01, -2.220095538485852e-01, -2.516748014421345e-01, -3.059006049095522e-01, -2.764666025884557e-01, -2.764666025884556e-01, -3.841933164150679e-01, -8.826678871922929e-02, -9.768823550019075e-02, -2.906360939159713e-01, -1.100793158919217e-01, -1.100793158919217e-01, -5.569847033060404e-02, -2.005383976925495e-11, -3.600759229185977e-10, -1.053619923927400e-01, -3.369326359045795e-09, -3.369326356908343e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hjs_b88_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_b88", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.931240669122806e+01, -1.931246641246774e+01, -1.931274804749799e+01, -1.931176979145796e+01, -1.931243805907578e+01, -1.931243805907578e+01, -3.218124896315546e+00, -3.218151483483197e+00, -3.219009876500607e+00, -3.218188026374403e+00, -3.218151779185213e+00, -3.218151779185213e+00, -6.722855515660944e-01, -6.711160167397667e-01, -6.392811359459962e-01, -6.446305974574725e-01, -6.718618676055041e-01, -6.718618676055041e-01, -2.023004514052058e-01, -2.065383957673448e-01, -8.062791471254156e-01, -1.346295419383542e-01, -2.035975299238770e-01, -2.035975299238770e-01, -2.461754488808984e-02, -2.420256271780822e-02, -5.387995403144447e-02, 1.388011792434954e-01, -2.397052228216670e-02, -2.397052228216670e-02, -4.849481114663227e+00, -4.851247341422373e+00, -4.849662252264065e+00, -4.851035579249636e+00, -4.850385938092259e+00, -4.850385938092259e+00, -1.768953051077257e+00, -1.783467943545574e+00, -1.760461583978721e+00, -1.771772339487530e+00, -1.789879129356859e+00, -1.789879129356859e+00, -6.178231636130481e-01, -6.754781315631257e-01, -5.888181099036131e-01, -6.205692323438701e-01, -6.410787015026946e-01, -6.410787015026946e-01, -9.813447503824904e-02, -1.858889759725681e-01, -9.718740923417964e-02, -1.890386740842812e+00, -1.134754638822837e-01, -1.134754638822837e-01, 1.710629872534189e-01, 3.579342987369764e-02, -2.515234289172461e-08, -6.695346911076637e-02, 3.062189461371925e-02, 3.062189461371776e-02, -6.442160379944464e-01, -6.418900698324053e-01, -6.429254744143791e-01, -6.435795681088072e-01, -6.432692118708294e-01, -6.432692118708294e-01, -6.252063876786857e-01, -5.328698288797938e-01, -5.638616217938428e-01, -5.905669023997515e-01, -5.773573980133058e-01, -5.773573980133058e-01, -7.048693012129834e-01, -2.381274185488221e-01, -2.916070832409041e-01, -3.874626043879170e-01, -3.377549688727721e-01, -3.377549688727719e-01, -4.854691447910842e-01, -5.203873471511798e-02, -6.275128078252773e-02, -3.762314829278205e-01, -8.206714714239245e-02, -8.206714714239216e-02, -2.599793281910964e-02, -1.321314906269836e-10, -2.363662832265945e-09, -7.884804758746995e-02, -2.205293940126641e-08, -2.205293939753708e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hjs_b88_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_b88", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.550447966091987e-09, -3.550425823231468e-09, -3.550284822944588e-09, -3.550648262065164e-09, -3.550436641599971e-09, -3.550436641599971e-09, -3.926619740249847e-06, -3.926570180352054e-06, -3.924590967423861e-06, -3.924865296436954e-06, -3.926531491510319e-06, -3.926531491510319e-06, -1.546486156778988e-03, -1.563384538186101e-03, -1.997423077932634e-03, -1.932625735694132e-03, -1.552598294883273e-03, -1.552598294883273e-03, -2.115127045760812e-01, -1.919085531598705e-01, -6.933258589287611e-04, -8.391631837803694e-01, -2.054340588421451e-01, -2.054340588421451e-01, -1.863115124719826e+03, -1.672026438910544e+03, -2.409658255789674e+01, -8.153394735882324e+04, -1.744532907221184e+03, -1.744532907221184e+03, -8.350765826477884e-07, -8.337636573968476e-07, -8.349431197540103e-07, -8.339224122325899e-07, -8.344055656612943e-07, -8.344055656612943e-07, -3.425906005864898e-05, -3.341254190602139e-05, -3.453500063044895e-05, -3.387071700018191e-05, -3.331769709428449e-05, -3.331769709428449e-05, -4.721304473888467e-04, 2.094204568533926e-03, -5.980407186075079e-04, 2.211440977894915e-03, -1.802470455783967e-04, -1.802470455783967e-04, -2.156061891961221e+00, -2.467903182048230e-01, -2.329386769452940e+00, -6.477536235748079e-06, -1.452542011759155e+00, -1.452542011759155e+00, -1.068705803099413e+05, -3.302004281960818e+04, 2.145382733561061e-02, -9.569246710977684e+00, -4.771694884961218e+04, -4.771694884961233e+04, 4.315038250307475e-03, 3.000179669127018e-03, 3.431368337939628e-03, 3.794644152076559e-03, 3.609622066930966e-03, 3.609622066930968e-03, 5.240018311230039e-03, -2.661688028236613e-03, -7.361431811920196e-04, 1.167974857304644e-03, 1.941328221803651e-04, 1.941328221803703e-04, 1.535137247932945e-03, -1.010198826124586e-01, -4.581140590436943e-02, -7.461615846768306e-03, -2.171853245794795e-02, -2.171853245794768e-02, -3.743287157298056e-03, -2.822882723534385e+01, -1.218842225335770e+01, 1.880019861392780e-03, -4.546252204491193e+00, -4.546252204491198e+00, -1.113975924739337e+03, 1.779839569319765e-03, 4.326607523398382e-03, -5.527583833885193e+00, 1.480927522127662e-02, 1.480927522044869e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05