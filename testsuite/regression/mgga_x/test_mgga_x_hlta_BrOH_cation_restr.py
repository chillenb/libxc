
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_hlta_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_hlta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.495940803029298e+01, -1.495926426597702e+01, -1.495871262574095e+01, -1.496083728357339e+01, -1.495971181752392e+01, -1.495971181752392e+01, -3.750254925531428e+00, -3.749870154691806e+00, -3.739798412027696e+00, -3.744206699902533e+00, -3.743344890576103e+00, -3.743344890576103e+00, -9.764883727678016e-01, -9.778732545998012e-01, -1.010448127955397e+00, -9.911608505447039e-01, -9.956804352881189e-01, -9.956804352881189e-01, -3.016799563614946e-01, -2.997274357442430e-01, -1.417797706202340e+00, -3.329938329301411e-01, -3.174674580987168e-01, -3.174674580987169e-01, -1.613358357920030e-01, -1.616539994060454e-01, -2.621723504065129e-01, -1.538250684053065e-01, -1.506961165539543e-01, -1.506961165539543e-01, -3.134408087829272e+00, -3.124882266796915e+00, -3.133855939340795e+00, -3.125452984943723e+00, -3.129640977015426e+00, -3.129640977015426e+00, -2.294143371875176e+00, -2.272795787665152e+00, -2.311982400703498e+00, -2.292327620548973e+00, -2.273396593418331e+00, -2.273396593418331e+00, -5.192562593127296e-01, -4.354458660659463e-01, -5.540203844137574e-01, -4.964813064345383e-01, -5.058180903350876e-01, -5.058180903350876e-01, -3.261995408568673e-01, -3.677424500242610e-01, -3.191025846751594e-01, -1.586783283466936e+00, -3.023383243495396e-01, -3.023383243495396e-01, -1.572100378130823e-01, -1.569540972290603e-01, -9.792631500631310e-02, -2.815452646447094e-01, -1.231025296874571e-01, -1.231025296874571e-01, -2.090572842052200e-01, -2.741590896929625e-01, -2.541575948736177e-01, -2.355448569538156e-01, -2.451365604714107e-01, -2.451365604714107e-01, -2.140844609537267e-01, -4.747986377854677e-01, -4.258812530178880e-01, -3.632582934530906e-01, -3.974752923492682e-01, -3.974752923492682e-01, -4.941175574000140e-01, -3.918314078371100e-01, -4.047374954498090e-01, -4.108848632458211e-01, -4.008108711179422e-01, -4.008108711179422e-01, -4.799320935490552e-01, -2.591676639970765e-01, -2.822578141687191e-01, -3.508708416437135e-01, -2.776909650135365e-01, -2.776909650135365e-01, -2.003341538181356e-01, -1.157751546021196e-01, -1.087712006277556e-01, -2.698914759524098e-01, -1.129139549904913e-01, -1.129139549904912e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_hlta_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_hlta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.972938686861989e+00, -9.972842843984681e+00, -9.972475083827302e+00, -9.973891522382260e+00, -9.973141211682611e+00, -9.973141211682611e+00, -2.500169950354285e+00, -2.499913436461204e+00, -2.493198941351797e+00, -2.496137799935023e+00, -2.495563260384069e+00, -2.495563260384069e+00, -6.509922485118679e-01, -6.519155030665341e-01, -6.736320853035979e-01, -6.607739003631358e-01, -6.637869568587460e-01, -6.637869568587460e-01, -2.011199709076631e-01, -1.998182904961620e-01, -9.451984708015601e-01, -2.219958886200941e-01, -2.116449720658112e-01, -2.116449720658113e-01, -1.075572238613353e-01, -1.077693329373636e-01, -1.747815669376753e-01, -1.025500456035377e-01, -1.004640777026362e-01, -1.004640777026362e-01, -2.089605391886181e+00, -2.083254844531277e+00, -2.089237292893864e+00, -2.083635323295816e+00, -2.086427318010283e+00, -2.086427318010283e+00, -1.529428914583451e+00, -1.515197191776768e+00, -1.541321600468999e+00, -1.528218413699315e+00, -1.515597728945554e+00, -1.515597728945554e+00, -3.461708395418198e-01, -2.902972440439643e-01, -3.693469229425049e-01, -3.309875376230255e-01, -3.372120602233916e-01, -3.372120602233916e-01, -2.174663605712448e-01, -2.451616333495073e-01, -2.127350564501063e-01, -1.057855522311291e+00, -2.015588828996930e-01, -2.015588828996930e-01, -1.048066918753882e-01, -1.046360648193735e-01, -6.528421000420874e-02, -1.876968430964728e-01, -8.206835312497140e-02, -8.206835312497140e-02, -1.393715228034800e-01, -1.827727264619750e-01, -1.694383965824118e-01, -1.570299046358770e-01, -1.634243736476071e-01, -1.634243736476071e-01, -1.427229739691511e-01, -3.165324251903118e-01, -2.839208353452586e-01, -2.421721956353937e-01, -2.649835282328454e-01, -2.649835282328454e-01, -3.294117049333427e-01, -2.612209385580734e-01, -2.698249969665393e-01, -2.739232421638808e-01, -2.672072474119614e-01, -2.672072474119614e-01, -3.199547290327032e-01, -1.727784426647176e-01, -1.881718761124795e-01, -2.339138944291423e-01, -1.851273100090243e-01, -1.851273100090244e-01, -1.335561025454237e-01, -7.718343640141304e-02, -7.251413375183706e-02, -1.799276506349398e-01, -7.527596999366085e-02, -7.527596999366082e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_hlta_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_hlta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_hlta_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_hlta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_hlta_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_hlta", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-5.847448831550570e-03, -5.847541280122541e-03, -5.847903516670066e-03, -5.846536480397518e-03, -5.847259506685833e-03, -5.847259506685833e-03, -1.879687143260924e-02, -1.879978489090543e-02, -1.887648333129324e-02, -1.884372691094293e-02, -1.884919420136088e-02, -1.884919420136088e-02, -6.213271836517701e-02, -6.196658664030944e-02, -5.820563942794758e-02, -6.016097950450481e-02, -5.969949509126601e-02, -5.969949509126601e-02, -1.838606575939554e-01, -1.870607956141999e-01, -3.834936967518026e-02, -1.381423190872139e-01, -1.572557783658942e-01, -1.572557783658942e-01, -8.771442047216083e-02, -8.972370096496116e-02, -1.020380363404856e-01, -7.160021649716868e-02, -8.277446713333830e-02, -8.277446713333830e-02, -3.010129902133910e-02, -3.024181967105902e-02, -3.010939226639910e-02, -3.023334954462686e-02, -3.017149941281894e-02, -3.017149941281894e-02, -2.947219443093752e-02, -2.999660299948526e-02, -2.905451262995713e-02, -2.952366103511680e-02, -2.997453455819527e-02, -2.997453455819527e-02, -1.498604942448535e-01, -2.033966683478858e-01, -1.308015643690469e-01, -1.572033216918958e-01, -1.571235237236754e-01, -1.571235237236754e-01, -1.187202285877802e-01, -1.367169503808549e-01, -1.180197380162820e-01, -5.074278465248532e-02, -1.441310883070598e-01, -1.441310883070598e-01, -6.087252235912439e-02, -6.868692283899556e-02, -1.218750272386401e-01, -1.167885739907685e-01, -9.490760691861978e-02, -9.490760691861977e-02, -5.880491370879836e-01, -3.904436852528536e-01, -4.378729652630199e-01, -4.911955206461481e-01, -4.624572044272016e-01, -4.624572044272015e-01, -5.597144067337945e-01, -1.570116715051141e-01, -1.891340926051724e-01, -2.453966344282994e-01, -2.120469944520125e-01, -2.120469944520125e-01, -1.722886435074615e-01, -1.397224326724265e-01, -1.458535291825365e-01, -1.638798107324618e-01, -1.586230465026387e-01, -1.586230465026388e-01, -1.482420786907230e-01, -1.015677837544208e-01, -1.043422473541989e-01, -2.039044258940660e-01, -1.345390936464649e-01, -1.345390936464649e-01, -7.530876531979233e-02, -5.605504115506192e-02, -8.918208941717153e-02, -1.360661366190060e-01, -1.039260318690901e-01, -1.039260318690901e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05