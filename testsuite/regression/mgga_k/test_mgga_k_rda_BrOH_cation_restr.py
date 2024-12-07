
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_rda_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.236913528534620e+03, 1.236909534939871e+03, 1.236911418513324e+03, 1.236970267134070e+03, 1.236936641439686e+03, 1.236936641439686e+03, 3.677146002386586e+01, 3.676853023608702e+01, 3.670269038481137e+01, "nan", 3.676972004967922e+01, "nan", 1.769201613993915e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", 2.256523593132327e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 6.793387522197318e+01, "nan", "nan", 1.893508606461987e+01, 1.892554645100804e+01, 1.938625151547703e+01, 1.937740246571522e+01, 1.871295600802654e+01, "nan", "nan", "nan", "nan", "nan", "nan", 9.819685871374698e-01, 6.707531401316820e-01, "nan", 6.750267735514389e-01, -1.613918183703985e+02, "nan", "nan", 6.671496312337629e-01, "nan", "nan", 6.909573741504096e-01, 4.125397375791776e-01, "nan", "nan", "nan", 7.881504803560798e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 7.814477484697311e-01, "nan", 5.709865430584696e-01, "nan", 5.120045984145961e-01, "nan", "nan", 8.066645531135113e-01, "nan", "nan", "nan", 5.455341583470822e-01, "nan", "nan", "nan", 4.581405237974066e-01, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_rda_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.083927157030925e+03, 1.083957643049796e+03, 1.084055847502057e+03, 1.083603001603139e+03, 1.083845589011651e+03, 1.083845589011651e+03, 2.094732705329914e+01, 2.095268792989872e+01, 2.107777561543980e+01, "nan", 2.095476479481268e+01, "nan", 2.619951717552340e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", 5.128808316107081e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 7.844372427310319e+01, "nan", "nan", -1.004091465060421e+00, -8.026890345387978e-01, -1.485272319319108e+00, -1.308460822835780e+00, -6.594762895147535e-01, "nan", "nan", "nan", "nan", "nan", "nan", 8.042803594559531e-01, -1.103576527950317e+00, "nan", -1.084189034116317e+00, 1.111152174065218e+02, "nan", "nan", -6.680039600929102e-01, "nan", "nan", -9.552821442527017e-01, -4.137020713885884e-01, "nan", "nan", "nan", 1.276716070541111e+00, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 5.977790958108896e-01, "nan", -5.085581166476210e-01, "nan", 3.637414127203626e-02, "nan", "nan", 1.210748797720131e-01, "nan", "nan", "nan", -8.623485835991077e-01, "nan", "nan", "nan", -4.585727352408373e-01, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_rda_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.475116343829923e-06, 2.475061305904150e-06, 2.474865275944198e-06, 2.475683203730591e-06, 2.475247722140933e-06, 2.475247722140933e-06, 6.455804523903605e-04, 6.455413491599557e-04, 6.445847647970178e-04, "nan", 6.454868857556694e-04, "nan", 9.582755621486416e-02, "nan", "nan", "nan", "nan", "nan", 1.212783639246232e+01, "nan", 6.030479441605948e-02, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.416141990659731e-04, "nan", "nan", 3.772979383617383e-03, 3.699878613365337e-03, 3.815087380847466e-03, 3.749284385743315e-03, 3.712603012812536e-03, "nan", "nan", "nan", "nan", "nan", "nan", 1.169907888992506e-01, 7.483648978444451e+01, "nan", 9.632210107954242e+01, 8.906969473199959e-03, "nan", "nan", 3.249256152706364e+06, "nan", "nan", 3.350456216341367e+02, 2.043736222100373e+06, "nan", "nan", "nan", 2.513546072983164e-02, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.735165829039715e-01, "nan", 3.079116283014677e+00, "nan", 6.858819259014296e-01, "nan", "nan", 3.104375898908925e-01, "nan", "nan", "nan", 1.548700144561999e+02, "nan", "nan", "nan", 9.043637243187241e+06, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_rda_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.142153570875613e-01, 1.142157151830822e-01, 1.142168221714479e-01, 1.142110387320254e-01, 1.142141580902706e-01, 1.142141580902706e-01, 1.093496080839404e-01, 1.093530802113637e-01, 1.094333345006684e-01, "nan", 1.093537710068463e-01, "nan", 9.863118280842345e-02, "nan", "nan", "nan", "nan", "nan", -1.456983634123979e-115, "nan", 1.013512225219895e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.179549840229342e-01, "nan", "nan", 8.835272910091677e-02, 8.905184863039430e-02, 8.674243107111061e-02, 8.737162796797003e-02, 8.951430006376279e-02, "nan", "nan", "nan", "nan", "nan", "nan", 1.133896387058127e-01, 7.243266892087820e-03, "nan", 5.911572627123751e-03, -4.097777573216461e-66, "nan", "nan", 1.780909359422850e-08, "nan", "nan", 1.867258449196394e-03, 8.624731482704104e-08, "nan", "nan", "nan", 1.249424992752386e-01, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 1.125817007239267e-01, "nan", 4.758313873901901e-02, "nan", 9.538348364845871e-02, "nan", "nan", 9.870691485768819e-02, "nan", "nan", "nan", 5.213131079255418e-03, "nan", "nan", "nan", -2.275800753451797e-72, "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_rda_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
