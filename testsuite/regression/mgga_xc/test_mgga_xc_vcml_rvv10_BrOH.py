
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_vcml_rvv10_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.178111324724589e+01, -2.178114123288091e+01, -2.178132999033965e+01, -2.178086961872585e+01, -2.178112748225454e+01, -2.178112748225454e+01, -3.424857572823093e+00, -3.424861058291753e+00, -3.425184084275983e+00, -3.426373110378790e+00, -3.424868023822770e+00, -3.424868023822770e+00, -6.914140829375394e-01, -6.910294225027737e-01, -6.827932977765121e-01, -6.886959805641206e-01, -6.912734688195773e-01, -6.912734688195773e-01, -2.225777649500006e-01, -2.230180270890384e-01, -8.072821088869726e-01, -1.512775588175455e-01, -2.224297265019905e-01, -2.224297265019905e-01, -6.472556415450334e-03, -6.790329773433226e-03, -3.345616930008893e-02, -3.060115219084154e-03, -6.720474970097323e-03, -6.720474970097323e-03, -5.286079070738895e+00, -5.285818152936915e+00, -5.286057822020288e+00, -5.285854859774323e+00, -5.285937312490145e+00, -5.285937312490145e+00, -2.202260565377533e+00, -2.215192581126625e+00, -2.204023868500149e+00, -2.212423373189959e+00, -2.209543430650045e+00, -2.209543430650045e+00, -6.252764547283686e-01, -6.793518807598263e-01, -5.784858587636903e-01, -6.044311526579943e-01, -6.552497759218292e-01, -6.552497759218292e-01, -1.015912245702289e-01, -2.246899935679369e-01, -1.012823829283731e-01, -1.901578287484361e+00, -1.269204265376870e-01, -1.269204265376870e-01, -2.951654984316999e-03, -3.376504865649094e-03, -2.530196507714730e-03, -5.385239895673634e-02, -3.076633055808115e-03, -3.076633055808113e-03, -6.436966826661090e-01, -6.414677471548753e-01, -6.422146213714269e-01, -6.428280211009278e-01, -6.425168120787484e-01, -6.425168120787484e-01, -6.253923743590181e-01, -5.634075136113176e-01, -5.800803381905644e-01, -5.962012400126393e-01, -5.879205137461767e-01, -5.879205137461767e-01, -7.025561349026143e-01, -2.921162926129964e-01, -3.267946227490926e-01, -3.827892077260124e-01, -3.548655936355025e-01, -3.548655936355023e-01, -5.051385441987792e-01, -2.976491520037413e-02, -4.532328423269562e-02, -3.613791992594951e-01, -7.998368040930300e-02, -7.998368040930300e-02, -7.606119180125561e-03, -9.210444590505242e-04, -1.751283649553526e-03, -7.640517498504644e-02, -2.613358148187333e-03, -2.613358148187333e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_vcml_rvv10_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.679790214903612e+01, -2.679796884351476e+01, -2.679819438847010e+01, -2.679709482400019e+01, -2.679793803723305e+01, -2.679793803723305e+01, -4.275492691076209e+00, -4.275697874898032e+00, -4.282196383840914e+00, -4.281120029588386e+00, -4.275562581638614e+00, -4.275562581638614e+00, -8.518294373178106e-01, -8.503566550720614e-01, -8.134258328380261e-01, -8.204506223122444e-01, -8.512962860399607e-01, -8.512962860399607e-01, -1.417634228171210e-01, -1.578407649867455e-01, -1.012490061695746e+00, -2.260213773909363e-01, -1.470198828121561e-01, -1.470198828121561e-01, -8.508789535831448e-03, -9.143942083496240e-03, -6.003266595726243e-02, -4.106724545111651e-03, -8.908357889019088e-03, -8.908357889019064e-03, -6.623228249089546e+00, -6.625705207382079e+00, -6.623440903470549e+00, -6.625369596691072e+00, -6.624552147891293e+00, -6.624552147891293e+00, -2.246009345247901e+00, -2.436921721260913e+00, -2.332400828798710e+00, -2.487903730062095e+00, -2.277965434049112e+00, -2.277965434049112e+00, -8.158747522410564e-01, -9.258123320684324e-01, -7.723353921068807e-01, -8.600188630503930e-01, -8.418431473642820e-01, -8.418431473642820e-01, -1.763305948149242e-01, -4.260881247106321e-02, -1.741234719039144e-01, -2.741437292667110e+00, -1.876404369520646e-01, -1.876404369520646e-01, -3.960089249587046e-03, -4.538857905205552e-03, -3.399799197171803e-03, -1.025240718690246e-01, -4.136066524862426e-03, -4.136066524862422e-03, -8.431613096183630e-01, -8.381315463739923e-01, -8.400799414181618e-01, -8.414830127156387e-01, -8.407933260894198e-01, -8.407933260894198e-01, -8.160588446574985e-01, -7.118276226973763e-01, -7.409034446261689e-01, -7.700424939226670e-01, -7.550144180795720e-01, -7.550144180795721e-01, -9.916281795808888e-01, -2.449662531130516e-01, -2.394995154764891e-01, -4.701298465330543e-01, -3.474841549524468e-01, -3.474841549524468e-01, -5.721454666176166e-01, -5.004723312065862e-02, -8.482674866745712e-02, -4.693814418619828e-01, -1.409600808711574e-01, -1.409600808711575e-01, -1.056759578906102e-02, -1.229033521676433e-03, -2.341264644989765e-03, -1.357113081601557e-01, -3.509557634842368e-03, -3.509557634842371e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_vcml_rvv10_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.395370423080191e-09, -4.395444891926979e-09, -4.396535147634544e-09, -4.395373378893036e-09, -4.395402423623706e-09, -4.395402423623706e-09, -7.245725334079383e-06, -7.251536300321455e-06, -7.437285147160975e-06, -7.471148606333423e-06, -7.246532852683186e-06, -7.246532852683186e-06, -3.137514908880024e-03, -3.154804660079710e-03, -3.555321019782747e-03, -3.607323918675358e-03, -3.143962558992664e-03, -3.143962558992664e-03, -3.481448732853880e+00, -3.182760540385090e+00, -1.238603294777907e-03, 2.519354907739436e-01, -3.387749083006595e+00, -3.387749083006595e+00, -3.868556559312566e+01, -1.453144400746070e+01, 5.661254659873433e+00, 9.001564460557571e+00, -3.094098377783661e+01, -3.094098377783534e+01, -1.622007623649811e-06, -1.621934595252338e-06, -1.622446687376586e-06, -1.622362156422254e-06, -1.621374362297889e-06, -1.621374362297889e-06, -1.225431910087793e-04, -5.513146261138844e-05, -8.538194968878440e-05, -3.300207093297960e-05, -1.198934357010483e-04, -1.198934357010483e-04, -2.147946884813853e-02, -3.744931893661591e-03, -1.781083227214561e-02, -1.449326317588460e-02, -1.601819873263865e-02, -1.601819873263865e-02, 9.203488480601208e-01, -3.312244117089521e+00, 9.459502936490641e-01, -1.332278959559722e-04, -6.735263336277011e-01, -6.735263336277011e-01, 9.536091778287497e+00, 9.607154535301470e+00, 2.737986626428110e+01, 4.712950690425799e+00, 1.419646559003505e+01, 1.419646559003510e+01, 1.896285599857305e-03, 7.239649522204870e-04, 1.092485518392182e-03, 1.414687886962591e-03, 1.250775439081255e-03, 1.250775439081243e-03, 6.006285836282014e-03, 3.603137716286097e-03, 2.566765691410308e-03, 2.351957583178167e-03, 2.455613501635734e-03, 2.455613501635737e-03, -6.825315954590125e-03, -6.615793464885610e-01, -6.929581388672094e-01, -1.232198113396413e-01, -3.778835163041918e-01, -3.778835163041915e-01, -8.237254875345892e-02, 3.002320074080903e+00, 4.339666581590152e+00, -1.435882807890548e-01, 1.569476743793320e+00, 1.569476743793346e+00, 1.025232248555931e+01, 1.621371656257687e+01, 1.398080315103217e+01, 2.245772643480750e+00, 2.068061797496303e+01, 2.068061797497101e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_vcml_rvv10_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_vcml_rvv10_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_vcml_rvv10", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.877006152698700e-04, -2.876863122211093e-04, -2.874904634970216e-04, -2.877158259927068e-04, -2.876943236658646e-04, -2.876943236658646e-04, 1.320516521641884e-03, 1.324543971750064e-03, 1.453572370468131e-03, 1.473120649500230e-03, 1.321195750823350e-03, 1.321195750823350e-03, 2.873628581160944e-03, 2.872026728817621e-03, 2.971356609416965e-03, 3.483069448787809e-03, 2.873626180684720e-03, 2.873626180684720e-03, 3.120670515585626e-01, 2.945634360570687e-01, 4.639973862403699e-04, 5.205842044696927e-03, 3.066519933533055e-01, 3.066519933533055e-01, 8.747181181628544e-04, 5.422382898235495e-04, 8.858386346478720e-04, 1.313795175731458e-09, 8.416596478846574e-04, 8.416596478846494e-04, -7.935239736686212e-05, -7.829172509282577e-05, -7.823466780181599e-05, -7.747225947829007e-05, -8.016414889003673e-05, -8.016414889003673e-05, 7.257311245941650e-03, -5.131435073854277e-04, 2.903634744972271e-03, -3.124731396226964e-03, 7.191347977352484e-03, 7.191347977352484e-03, 5.942089614958538e-02, 2.892369377391261e-02, 4.093412104368779e-02, 5.253788154856460e-02, 4.529412342816947e-02, 4.529412342816947e-02, 7.453765831020137e-03, 3.224566236842586e-01, 8.574279482657394e-03, 1.526790055178626e-02, 3.464213790962529e-02, 3.464213790962529e-02, 1.250160858121261e-08, 5.989180652797596e-09, 1.083992381421028e-07, 2.671933327264298e-04, 2.518421467135024e-09, 2.518421466665899e-09, -1.221832832301905e-02, -1.172409322031500e-02, -1.186436821216011e-02, -1.199903577900436e-02, -1.193375750008194e-02, -1.193375750008194e-02, -2.311539269526043e-02, -2.685552237794925e-02, -2.386988664322194e-02, -2.124156122755433e-02, -2.280771357421443e-02, -2.280771357421445e-02, 4.634998556275245e-02, 1.118684659089915e-01, 2.014252137153657e-01, 6.784102322621473e-02, 1.521347735751405e-01, 1.521347735751404e-01, 1.010515571748664e-01, 2.598365560162005e-03, 1.163576379164968e-03, 7.305926754999666e-02, 1.071240358842291e-02, 1.071240358842275e-02, 1.473642764907289e-07, 2.561015037738952e-12, 1.816163802813239e-08, 9.193235006593232e-03, 1.645459816663567e-09, 1.645459817309182e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05