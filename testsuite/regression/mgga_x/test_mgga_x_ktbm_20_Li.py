
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_20_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.000908705507028e+00, -1.308507012505415e+00, -2.610540578946680e-01, -1.852308760998566e-01, -5.529549302069900e-02, -1.065101226805570e-02, -1.998910425209877e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_20_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.871424781007129e+00, -2.874230564745265e+00, -1.986442461101691e+00, -1.988262611333175e+00, -3.327174758042768e-01, -3.324037317980454e-01, -2.588637387719755e-01, -1.331615557021979e-02, -7.461455745022907e-02, -4.222849573986462e-04, -1.400147595542929e-02, -1.390108490441107e-02, -2.820620218046046e-04, -2.092350908620581e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_20_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.779890286385669e-04, 0.000000000000000e+00, -7.753927035852195e-04, -2.812227688409718e-03, 0.000000000000000e+00, -2.805196515185624e-03, -2.699383513902266e-02, 0.000000000000000e+00, -2.878345617073836e-02, -1.235579729308379e+01, 0.000000000000000e+00, -6.154270755122951e+00, -6.275980369302238e+01, 0.000000000000000e+00, -1.530778014165164e+04, -1.136701533510975e-01, 0.000000000000000e+00, -5.504988024685737e+00, -2.317994097191974e-01, 0.000000000000000e+00, 2.328990277244818e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_20_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.920981733504290e-02, 2.918267759213949e-02, 3.548818548117648e-02, 3.549461029499876e-02, -2.830291510150826e-03, -2.877089949238418e-03, 3.360324811739620e-01, 7.899610796460225e-05, 7.428673246963478e-03, 6.237012245188106e-06, 1.688330598181494e-06, 8.041179621214504e-05, 2.814416972746813e-11, -9.887823405560174e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
