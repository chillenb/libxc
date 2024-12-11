
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lambda_oc2_n_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_oc2_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.793507287810147e+00, -1.281337805614602e+00, -3.948742392563092e-01, -1.599955848476881e-01, -7.833153165468551e-02, -1.808207844423949e-02, -3.377704096037725e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lambda_oc2_n_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_oc2_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.245915332072631e+00, -2.248040999810667e+00, -1.526908291933804e+00, -1.528268465629625e+00, -4.197080137528711e-01, -4.198326030879718e-01, -2.054187191272109e-01, -2.299513223604202e-02, -8.041785535316721e-02, -7.300326318288619e-04, -2.417815020457842e-02, -2.400358540514355e-02, -4.876206349549359e-04, -3.466538542970280e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lambda_oc2_n_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_oc2_n", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.474702417534440e-04, 0.000000000000000e+00, -2.466309173867137e-04, -9.528683700642288e-04, 0.000000000000000e+00, -9.498701857517464e-04, -5.180511919451224e-02, 0.000000000000000e+00, -5.164710258153508e-02, -3.879053241829765e+00, 0.000000000000000e+00, -1.483370465124468e-01, -5.257055286622798e+01, 0.000000000000000e+00, -9.482257070913590e-01, -1.507536091046070e-01, 0.000000000000000e+00, -1.407722786190895e-01, -6.902726283951930e-01, 0.000000000000000e+00, -9.880539963625846e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
