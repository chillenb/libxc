
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpwlyp1m_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpwlyp1m", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.768392152343264e+00, -1.276453042843348e+00, -3.914952780273839e-01, -1.525732142329866e-01, -7.697295180724339e-02, -3.704660404482424e-03, -3.010002891240523e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpwlyp1m_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpwlyp1m", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.193877390823108e+00, -2.195722521567383e+00, -1.513573700797752e+00, -1.514696021727770e+00, -4.454615902681910e-01, -4.458711946247024e-01, -1.948046895580187e-01, -8.753332399848561e-02, -7.088458075509518e-02, -3.030037381148620e-02, -8.823126140635141e-03, -8.614259007607651e-03, -2.106080563777447e-05, -9.372999341721871e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpwlyp1m_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpwlyp1m", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.556589248643381e-04, 5.222815421851711e-06, -2.549180164026762e-04, -9.411482413550912e-04, 3.646941789248587e-05, -9.387391910558496e-04, -6.223721723476577e-02, 4.773762863586187e-02, -6.199500301516611e-02, -4.232973233124191e+00, 4.596134769453040e+00, 3.594900430165508e+01, -6.946772197256679e+01, 2.356939734329661e+01, 4.556179563475872e+02, 3.262760733690252e+01, 7.936097321777658e-02, 3.065688512558664e+01, 3.637149712967952e+02, 0.000000000000000e+00, 5.578937441182695e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
