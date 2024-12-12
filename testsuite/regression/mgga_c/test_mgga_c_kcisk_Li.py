
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_kcisk_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.024502343860101e-02, -3.331800854853425e-02, -1.216841656782371e-02, -8.245360362239227e-04, -6.204628358862052e-09, -4.353984624350511e-05, -2.840259883800718e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_kcisk_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.579328328329433e-02, -8.561250601843158e-02, -7.428065028082070e-02, -7.411779092712942e-02, -3.488891840985017e-02, -3.469430019894524e-02, -1.691003994988853e-02, -1.123901976757180e-01, -4.952096224784564e-03, -5.233358645713876e-02, -1.724467767131775e-04, -1.531054523049274e-04, -9.717196170275458e-08, -1.278783889833075e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.096019274048747e-04, 7.320042881194598e-05, 1.095204911247090e-04, 2.577887103951170e-04, 2.271129490296613e-04, 2.576477640625128e-04, 7.322645328039913e-03, 1.268765796888489e-02, 7.343289274288431e-03, 2.058632499967424e+01, 3.448887585357340e+00, 1.725008668310976e+00, 2.889686716710776e+01, 3.073569834891379e+01, 1.537443133179414e+01, 5.210445871476178e-01, 1.042065283922323e+00, 5.215590871661558e-01, 1.321752999200642e+02, 2.643505997762938e+02, 1.321821336856628e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-5.949321698326564e-03, -5.959763386341452e-03, -4.513505340771390e-03, -4.525576659408382e-03, -1.164494420876792e-03, -1.224517918735917e-03, -7.493133351448974e-01, -2.443810853658494e-06, -6.910616689774313e-02, -1.572871022272044e-08, -1.148406673888586e-09, -2.527468185318665e-06, -3.203183121790221e-19, -3.695772064639163e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
