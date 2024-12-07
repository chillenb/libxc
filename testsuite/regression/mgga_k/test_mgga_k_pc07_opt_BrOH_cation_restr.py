
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_pc07_opt_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07_opt", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.783120983392261e+03, 1.783132191103062e+03, 1.783183121926190e+03, 1.783016061520877e+03, 1.783103448629590e+03, 1.783103448629590e+03, 4.731362970739786e+01, 4.731331436318954e+01, 4.730860893569967e+01, "nan", 4.731553282722619e+01, "nan", 2.195201170066716e+00, "nan", "nan", "nan", "nan", "nan", 6.132771423699879e-01, "nan", 2.693956637184296e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.070396323720135e+02, "nan", "nan", 2.720444925399673e+01, 2.696575746413651e+01, 2.837135809099123e+01, 2.815777963975927e+01, 2.651392869737744e+01, "nan", "nan", "nan", "nan", "nan", "nan", 1.387351760730489e+00, 9.219337397706971e-01, "nan", 8.996643050506318e-01, 1.748435613897077e+01, "nan", "nan", 6.678040803851272e-01, "nan", "nan", 8.107085906494392e-01, 4.134218398231095e-01, "nan", "nan", "nan", 1.397675225853993e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.082629239099385e+00, "nan", 1.064541222388073e+00, "nan", 6.698639965281545e-01, "nan", "nan", 9.995766271833780e-01, "nan", "nan", "nan", 7.143488347941436e-01, "nan", "nan", "nan", 4.583175932800985e-01, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_pc07_opt_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07_opt", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.087054293052819e+03, 3.087084545002082e+03, 3.087203464491818e+03, 3.086753453290542e+03, 3.086991514210636e+03, 3.086991514210636e+03, 7.206390566310614e+01, 7.207696603030024e+01, 7.237916619072352e+01, "nan", 7.208303819643757e+01, "nan", -2.175100688880093e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", -7.239323004094713e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.875867705193157e+02, "nan", "nan", -2.720444925399673e+01, -2.696575746413649e+01, -2.837135809099123e+01, -2.815777963975924e+01, -2.651392869737742e+01, "nan", "nan", "nan", "nan", "nan", "nan", 2.380381340480857e+00, -9.219337397706977e-01, "nan", -8.996643050506312e-01, -4.922297305660642e+01, "nan", "nan", -6.669284906701317e-01, "nan", "nan", -8.107085906494390e-01, -4.122777827275363e-01, "nan", "nan", "nan", 2.381770615624848e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.834798657857998e+00, "nan", -1.064541222388072e+00, "nan", -6.698639965281546e-01, "nan", "nan", -9.866462676224121e-01, "nan", "nan", "nan", -7.143488347941434e-01, "nan", "nan", "nan", -4.582760094805206e-01, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_opt_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07_opt", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.916327265768193e-07, -2.916665306678413e-07, -2.917579680346895e-07, -2.912556675384146e-07, -2.915273448116572e-07, -2.915273448116572e-07, 1.087021544855526e-04, 1.085057075092222e-04, 1.039948982811636e-04, "nan", 1.084587959173850e-04, "nan", 2.080772797923116e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", 9.680466400205365e-02, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", -3.740903744654567e-05, "nan", "nan", 8.405710951977128e-03, 8.225420308312626e-03, 8.540671341115710e-03, 8.377662191483195e-03, 8.242192632024158e-03, "nan", "nan", "nan", "nan", "nan", "nan", -9.575959184219558e-03, 8.282011699658828e+01, "nan", 1.046004417152880e+02, -2.314455549617628e-03, "nan", "nan", 3.249284044922582e+06, "nan", "nan", 3.437938582102904e+02, 2.043821369548235e+06, "nan", "nan", "nan", -3.566457619151864e-02, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", -7.490107755901924e-03, "nan", 5.986104049931646e+00, "nan", 1.499685834607407e+00, "nan", "nan", 6.731174422249531e-01, "nan", "nan", "nan", 1.665281126261038e+02, "nan", "nan", "nan", 9.043684888226071e+06, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_opt_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07_opt", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.956516123343247e-01, 1.956526709273287e-01, 1.956556805590475e-01, 1.956399497390609e-01, 1.956484364891294e-01, 1.956484364891294e-01, 1.735194216481251e-01, 1.735439438074664e-01, 1.741061573640076e-01, "nan", 1.735488206291535e-01, "nan", 6.371693119819494e-04, "nan", "nan", "nan", "nan", "nan", 0.000000000000000e+00, "nan", 4.950853730860030e-02, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 2.029926986308444e-01, "nan", "nan", 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, "nan", "nan", "nan", "nan", "nan", "nan", 1.932175236859447e-01, 0.000000000000000e+00, "nan", 0.000000000000000e+00, 8.483723295432121e-67, "nan", "nan", 8.761498735157847e-04, "nan", "nan", 0.000000000000000e+00, 1.910114831456515e-03, "nan", "nan", "nan", 2.089326384705030e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.904247257135677e-01, "nan", 0.000000000000000e+00, "nan", 0.000000000000000e+00, "nan", "nan", 8.995712753265295e-04, "nan", "nan", "nan", 0.000000000000000e+00, "nan", "nan", "nan", -3.712249139123396e-58, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_opt_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07_opt", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
