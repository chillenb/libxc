
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw1pw_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1pw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.587175204570227e+01, -1.587177280418768e+01, -1.587191613118015e+01, -1.587160623427432e+01, -1.587176135570810e+01, -1.587176135570810e+01, -2.671180750167573e+00, -2.671156450861577e+00, -2.670659255480212e+00, -2.672022851929786e+00, -2.671214949595042e+00, -2.671214949595042e+00, -5.565556644283710e-01, -5.563365985522186e-01, -5.523689076651025e-01, -5.562414183972799e-01, -5.549903951574530e-01, -5.549903951574530e-01, -1.703452310275626e-01, -1.715549710941005e-01, -6.406705132124551e-01, -1.449098736461730e-01, -1.537560634402414e-01, -1.537560634402415e-01, -6.408859462286842e-04, -7.563452685499779e-04, -4.172424872951034e-02, -1.347575469870231e-04, -2.814401513141034e-04, -2.814401513141032e-04, -3.863115093517138e+00, -3.862762385664876e+00, -3.863106446752579e+00, -3.862794954055053e+00, -3.862932060230272e+00, -3.862932060230272e+00, -1.621384563026124e+00, -1.629363799541614e+00, -1.621194478673683e+00, -1.628226960567007e+00, -1.625900358672477e+00, -1.625900358672477e+00, -4.786840317223092e-01, -5.099830279446721e-01, -4.466255278074294e-01, -4.564891038609210e-01, -4.847320536515504e-01, -4.847320536515504e-01, -1.174923768560136e-01, -1.818653273236897e-01, -1.108407387439219e-01, -1.438032661126698e+00, -1.260137018373448e-01, -1.260137018373448e-01, -7.439834359992793e-05, -1.359926066976506e-04, -1.455986289334960e-04, -7.945509597228879e-02, -1.738566175269394e-04, -1.738566175269396e-04, -4.740983478609709e-01, -4.721290930618819e-01, -4.727909286653694e-01, -4.733719116403765e-01, -4.730778228068774e-01, -4.730778228068774e-01, -4.621750624002238e-01, -4.145028486707028e-01, -4.265103032100165e-01, -4.393497047507619e-01, -4.325796485409528e-01, -4.325796485409528e-01, -5.328603808418452e-01, -2.161431408863859e-01, -2.456602681082919e-01, -3.009474658203680e-01, -2.705705802094396e-01, -2.705705802094395e-01, -3.839006321449523e-01, -3.792179354949317e-02, -6.176696391217485e-02, -2.878071429194216e-01, -9.599795381243489e-02, -9.599795381243488e-02, -1.506913586044464e-03, -7.239649440041040e-06, -4.042527716475956e-05, -9.174962963475275e-02, -1.555069367619319e-04, -1.555069367619286e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw1pw_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1pw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.904536673511987e+01, -1.904543815172493e+01, -1.904574201254440e+01, -1.904467976800400e+01, -1.904523829688445e+01, -1.904523829688445e+01, -3.155536819128111e+00, -3.155569899117824e+00, -3.156410181598614e+00, -3.155530476458756e+00, -3.155650157857285e+00, -3.155650157857285e+00, -6.545892073542349e-01, -6.534651363861006e-01, -6.257744574448354e-01, -6.310545764798821e-01, -6.303946348558375e-01, -6.303946348558375e-01, -1.748806544802277e-01, -1.786845895473368e-01, -7.574651000442165e-01, -1.296633102088070e-01, -1.456238653478270e-01, -1.456238653478270e-01, -2.359045238933788e-03, -2.781809782022773e-03, -9.433865470052667e-02, -4.957997357569322e-04, -1.036890895467433e-03, -1.036890895467439e-03, -4.781106574484127e+00, -4.783131769428683e+00, -4.781197442394968e+00, -4.782985289329858e+00, -4.782134037544902e+00, -4.782134037544902e+00, -1.740237582549560e+00, -1.754349035170645e+00, -1.726924421503206e+00, -1.739331937484902e+00, -1.754560492212407e+00, -1.754560492212407e+00, -5.989688764567208e-01, -6.625765092322942e-01, -5.558401548236345e-01, -5.909355323225289e-01, -6.100144124699531e-01, -6.100144124699531e-01, -1.086352992049781e-01, -1.678365017467242e-01, -1.059164940439917e-01, -1.863469522043038e+00, -1.103150062319111e-01, -1.103150062319111e-01, -2.733523369701768e-04, -5.004028805116399e-04, -5.362936633414217e-04, -1.005599724333589e-01, -6.403657213075948e-04, -6.403657213075961e-04, -6.180171537465554e-01, -6.136984556206313e-01, -6.155711982834391e-01, -6.168538851684873e-01, -6.162502073537279e-01, -6.162502073537279e-01, -6.027963984499107e-01, -4.971834349787442e-01, -5.290848218540699e-01, -5.607594224466491e-01, -5.447769606588826e-01, -5.447769606588826e-01, -6.914913066245219e-01, -2.131852021647994e-01, -2.623621832024701e-01, -3.628083860853709e-01, -3.096340708951282e-01, -3.096340708951282e-01, -4.613772862615550e-01, -9.119875221167752e-02, -1.076175760936319e-01, -3.580372162312917e-01, -9.402457622538506e-02, -9.402457622538503e-02, -5.510494922683132e-03, -2.646374269937859e-05, -1.483554171024137e-04, -9.424705216938074e-02, -5.727332440388077e-04, -5.727332440388001e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw1pw_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1pw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.307629070118672e-09, -5.307601253284533e-09, -5.307409804660095e-09, -5.307825066331373e-09, -5.307617109931044e-09, -5.307617109931044e-09, -6.264266271780590e-06, -6.264407946223566e-06, -6.267084058764054e-06, -6.257831037786598e-06, -6.263852707028584e-06, -6.263852707028584e-06, -2.818268910339717e-03, -2.830670991585210e-03, -3.110965232357395e-03, -3.020781647458287e-03, -3.041342540635445e-03, -3.041342540635445e-03, -3.735116289222830e-01, -3.538774747207117e-01, -1.623766415323911e-03, -8.538144188280153e-01, -6.333258003929433e-01, -6.333258003929433e-01, 2.223191985312352e+02, 2.310272318108772e+02, 2.321011741799631e+01, 2.309521390920983e+02, 2.746928329037112e+02, 2.746928329037145e+02, -1.415372899960643e-06, -1.415104131416298e-06, -1.415353548354859e-06, -1.415116580065744e-06, -1.415242205776729e-06, -1.415242205776729e-06, -4.792688629659729e-05, -4.689997666373255e-05, -4.820379375626732e-05, -4.728999177697080e-05, -4.722266947048051e-05, -4.722266947048051e-05, -3.898523501565928e-03, -1.336039511836506e-03, -5.220500548839841e-03, -2.379087337169728e-03, -3.554453311943177e-03, -3.554453311943177e-03, -1.613738407653315e+00, -3.342480437284432e-01, -1.821634882921519e+00, -4.832655863578080e-05, -1.489407457793622e+00, -1.489407457793622e+00, 3.179203405820148e+02, 2.706929711141524e+02, 1.627417802451359e+03, -8.522264611249706e-01, 7.582009793741466e+02, 7.582009793741461e+02, -1.133745384149100e-03, -1.531574782123644e-03, -1.176185886713879e-03, -9.798593636713097e-04, -1.056349113465029e-03, -1.056349113465029e-03, -8.137389681106798e-04, -8.200595391001301e-03, -6.314985420175126e-03, -4.220183234867518e-03, -5.323430167541113e-03, -5.323430167541113e-03, -1.295053278148903e-03, -1.543010481036831e-01, -8.179611920977137e-02, -2.675806699469841e-02, -4.810618717528389e-02, -4.810618717528393e-02, -1.087399764049049e-02, 2.642551949531336e+01, 6.645842693607636e+00, -2.581417300319660e-02, -2.963457981548228e+00, -2.963457981548228e+00, 1.654019175586454e+02, 1.653504119155636e+03, 6.989518736782410e+02, -2.913578340263367e+00, 9.619004576415870e+02, 9.619004576415750e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05