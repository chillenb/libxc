
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sfat_pbe_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.071087086165877e+01, -2.071089878819841e+01, -2.071108986463513e+01, -2.071067299711534e+01, -2.071088190901046e+01, -2.071088190901046e+01, -3.246059326543101e+00, -3.246025649525525e+00, -3.245332022092889e+00, -3.247195117226752e+00, -3.246102340020023e+00, -3.246102340020023e+00, -5.016216588143289e-01, -5.013801766604072e-01, -4.965298417269072e-01, -5.013258567077032e-01, -4.998023421072099e-01, -4.998023421072099e-01, -6.636747578620944e-02, -6.799826947708090e-02, -6.056995708372704e-01, -3.931691213077437e-02, -4.939620043542896e-02, -4.939620043542896e-02, -7.042147357312858e-06, -8.209836149381493e-06, -1.299665558400834e-03, -1.358241593183885e-06, -2.693926389304939e-06, -2.693926389304939e-06, -4.813340797339000e+00, -4.812821606852744e+00, -4.813327183321483e+00, -4.812868682627747e+00, -4.813071921327248e+00, -4.813071921327248e+00, -1.870765671120002e+00, -1.881319176242982e+00, -1.870193925682633e+00, -1.879531733349594e+00, -1.876854091845522e+00, -1.876854091845522e+00, -4.033449633804702e-01, -4.330681281158289e-01, -3.656940875992116e-01, -3.707506940681286e-01, -4.099184657478943e-01, -4.099184657478943e-01, -1.731311172967631e-02, -6.896791344092341e-02, -1.429266258840731e-02, -1.613415431500564e+00, -2.508699763459151e-02, -2.508699763459151e-02, -6.240332935367337e-07, -1.268987495465653e-06, -5.674799846170691e-07, -5.066083666548508e-03, -9.919905088838872e-07, -9.919905088838872e-07, -3.868131178243403e-01, -3.875663685574178e-01, -3.873146529904266e-01, -3.870916258401645e-01, -3.872037827449104e-01, -3.872037827449104e-01, -3.718654747693409e-01, -3.298124218245811e-01, -3.422210695567377e-01, -3.539367725376863e-01, -3.478931419389024e-01, -3.478931419389024e-01, -4.611792329976066e-01, -1.043707175565041e-01, -1.384481350268877e-01, -2.004032818728517e-01, -1.672303037005334e-01, -1.672303037005334e-01, -2.940333176792193e-01, -1.145261949440410e-03, -2.773915245186946e-03, -1.852605509457104e-01, -9.613929032949520e-03, -9.613929032949522e-03, -1.975876075697316e-05, -2.425090511242744e-08, -2.242482966873606e-07, -8.163329625955097e-03, -7.859684116583962e-07, -7.859684116583942e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sfat_pbe_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.495331638585279e+01, -2.495341547629339e+01, -2.495383116820737e+01, -2.495235742385657e+01, -2.495313316241666e+01, -2.495313316241666e+01, -3.861843226927832e+00, -3.861884264153387e+00, -3.862932638844346e+00, -3.861876463981586e+00, -3.861989592705710e+00, -3.861989592705710e+00, -6.160491208118610e-01, -6.150775715616694e-01, -5.930507100344649e-01, -5.990318748851593e-01, -5.978004486615662e-01, -5.978004486615662e-01, -9.658580241649847e-02, -9.850525627989247e-02, -7.417968868595045e-01, -6.269078496815013e-02, -7.556961429125450e-02, -7.556961429125450e-02, -1.407543816831771e-05, -1.640823317396152e-05, -2.546072818542193e-03, -2.715913294607865e-06, -5.386067872583780e-06, -5.386067872583780e-06, -6.008453162876303e+00, -6.011075887950621e+00, -6.008571110837558e+00, -6.010886482241792e+00, -6.009783863299170e+00, -6.009783863299170e+00, -2.040289429224792e+00, -2.056892111724244e+00, -2.027320632935592e+00, -2.041786253909762e+00, -2.055968986053542e+00, -2.055968986053542e+00, -5.315773410414755e-01, -6.035972956597337e-01, -4.807737228749387e-01, -5.163464314625709e-01, -5.437463265075176e-01, -5.437463265075176e-01, -3.060777290285769e-02, -1.018959274246069e-01, -2.564697743065482e-02, -2.141124674236901e+00, -4.238092192976711e-02, -4.238092192976711e-02, -1.247910688871120e-06, -2.537466102157171e-06, -1.134826787505885e-06, -9.604660872053526e-03, -1.983643250266568e-06, -1.983643250266568e-06, -5.520550190184066e-01, -5.440989090530375e-01, -5.468471410706559e-01, -5.491576826974428e-01, -5.479971406169203e-01, -5.479971406169203e-01, -5.344733405031641e-01, -4.202314826251611e-01, -4.495378872677568e-01, -4.815821687177815e-01, -4.649356340408313e-01, -4.649356340408313e-01, -6.390145639908305e-01, -1.439151388185164e-01, -1.829849285639438e-01, -2.644568259930183e-01, -2.186936274708418e-01, -2.186936274708418e-01, -3.776446754282758e-01, -2.247544807504372e-03, -5.359267188854461e-03, -2.520325755972555e-01, -1.763951780182436e-02, -1.763951780182434e-02, -3.946807833116420e-05, -4.850111521178075e-08, -4.484682753578049e-07, -1.512497360672020e-02, -1.571707669169786e-06, -1.571707669169782e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sfat_pbe_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.933161565301722e-09, -6.933113648328335e-09, -6.932830171907398e-09, -6.933544531679706e-09, -6.933180233622822e-09, -6.933180233622822e-09, -8.599714330442359e-06, -8.599980843008637e-06, -8.605317944223941e-06, -8.589670503663059e-06, -8.599224990278663e-06, -8.599224990278663e-06, -3.698161532758238e-03, -3.697905863289030e-03, -3.639815463621721e-03, -3.555539485012187e-03, -3.589492189768287e-03, -3.589492189768287e-03, -9.902303841874344e-02, -1.019698828609074e-01, -2.213495191040873e-03, -7.296598252649020e-02, -9.078633248728098e-02, -9.078633248728089e-02, -3.201853706892149e-06, -4.149877796216659e-06, -1.871051714527920e-03, -3.245987389888868e-07, -1.027407842016649e-06, -1.027407842016544e-06, -1.941244154524072e-06, -1.941487059983948e-06, -1.941242301981500e-06, -1.941456949646851e-06, -1.941374628706551e-06, -1.941374628706551e-06, -6.107849193087810e-05, -5.998860776685018e-05, -6.094607422858822e-05, -5.999432551921106e-05, -6.052729678718646e-05, -6.052729678718646e-05, -6.973156519197011e-03, -5.970696303866836e-03, -8.900260710843752e-03, -8.977459089603080e-03, -6.722391742960338e-03, -6.722391742960338e-03, -3.229157400808186e-02, -5.835972654031209e-02, -2.848835700890398e-02, -1.003941217223989e-04, -6.105114641688793e-02, -6.105114641688793e-02, -1.516696225499311e-07, -3.489466149992027e-07, -7.498261737556094e-07, -1.198471638964103e-02, -7.345972329458382e-07, -7.345972329462357e-07, -8.133189902003036e-03, -8.036262768110473e-03, -8.070194940619183e-03, -8.098624692982071e-03, -8.084428105118991e-03, -8.084428105118991e-03, -9.040156041820643e-03, -1.090485815065721e-02, -1.044926429404963e-02, -9.945133236576672e-03, -1.022221015083501e-02, -1.022221015083501e-02, -5.024451345416560e-03, -5.045346524188921e-02, -4.473963102010506e-02, -3.236708534650076e-02, -3.988240862482088e-02, -3.988240862482088e-02, -1.427876176962653e-02, -1.447098066955158e-03, -4.491971159249758e-03, -4.061574170145471e-02, -2.952444243962873e-02, -2.952444243962865e-02, -1.019195881212025e-05, -8.753928154139355e-09, -8.222050667806774e-08, -2.441999433600175e-02, -6.803302496398917e-07, -6.803302496397941e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05