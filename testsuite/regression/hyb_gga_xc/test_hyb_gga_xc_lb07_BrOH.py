
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lb07_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lb07", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.765079611403009e+01, -1.765083949252483e+01, -1.765106479325076e+01, -1.765035381926709e+01, -1.765081872473040e+01, -1.765081872473040e+01, -2.715718072674338e+00, -2.715724911488969e+00, -2.716032338919343e+00, -2.716117789944274e+00, -2.715733953913933e+00, -2.715733953913933e+00, -3.792033325302031e-01, -3.785517145389955e-01, -3.611431316334677e-01, -3.653979391392412e-01, -3.789662574988480e-01, -3.789662574988480e-01, -3.451961127693130e-02, -3.629724911035665e-02, -4.883833874238984e-01, -5.367110975220506e-03, -3.506568853738842e-02, -3.506568853738842e-02, -1.745439028464953e-03, -1.825780024176483e-03, 2.386578448605804e-03, -8.508172693810730e-04, -1.807319277183199e-03, -1.807319277183199e-03, -4.159598434815143e+00, -4.160389221017236e+00, -4.159681771194814e+00, -4.160296617117957e+00, -4.160000509854551e+00, -4.160000509854551e+00, -1.438890786435397e+00, -1.450631578867648e+00, -1.433868804107971e+00, -1.443039585521400e+00, -1.453323026194967e+00, -1.453323026194967e+00, -3.156849800292727e-01, -3.583191417863465e-01, -2.922138307582771e-01, -3.135018461418118e-01, -3.337746455699711e-01, -3.337746455699711e-01, 1.075832697642039e-02, -2.440646681849171e-02, 8.789855659326763e-03, -1.418516345852004e+00, 2.391147457473504e-03, 2.391147457473504e-03, -8.212295463803766e-04, -9.365065417081290e-04, -7.049864794051610e-04, 7.504339637310757e-03, -8.546104734814601e-04, -8.546104734814601e-04, -3.340892538593169e-01, -3.309410460332462e-01, -3.320704802948367e-01, -3.329415854168105e-01, -3.325057671832096e-01, -3.325057671832096e-01, -3.190547039922382e-01, -2.525737207891695e-01, -2.721999374553200e-01, -2.902376291418209e-01, -2.810902941521618e-01, -2.810902941521618e-01, -3.825684367389771e-01, -5.210330673532423e-02, -8.258721206585994e-02, -1.400902443270975e-01, -1.095152214147552e-01, -1.095152214147551e-01, -2.147803964507427e-01, 1.807910758309276e-03, 7.588218367983975e-03, -1.291824859227020e-01, 7.519017817532434e-03, 7.519017817532424e-03, -2.035484421017095e-03, -2.597660170533081e-04, -4.912165285735281e-04, 5.575763561159937e-03, -7.280461888973606e-04, -7.280461888973595e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lb07_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lb07", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.360515020167430e+01, -2.360520780940571e+01, -2.360550751571556e+01, -2.360456331456331e+01, -2.360518022506230e+01, -2.360518022506230e+01, -3.695307256741092e+00, -3.695314110222005e+00, -3.695661854217062e+00, -3.695883086314127e+00, -3.695327269169061e+00, -3.695327269169061e+00, -5.805889871766802e-01, -5.800107717857328e-01, -5.659088319271296e-01, -5.711905016369667e-01, -5.803771253717404e-01, -5.803771253717404e-01, -9.299513605807611e-02, -9.396603857204133e-02, -7.218022592722977e-01, -6.954901798668911e-02, -9.326677980769846e-02, -9.326677980769846e-02, -2.310579334045723e-03, -2.416259941995268e-03, -1.042983764727815e-02, -1.129960132027580e-03, -2.391988942889066e-03, -2.391988942889066e-03, -5.614273182909799e+00, -5.615273405252349e+00, -5.614378880481160e+00, -5.615156560128425e+00, -5.614781316925860e+00, -5.614781316925860e+00, -2.013398954849052e+00, -2.028252722660900e+00, -2.007863130983202e+00, -2.019442867439454e+00, -2.030650654622793e+00, -2.030650654622793e+00, -4.785600354334122e-01, -5.280611187081643e-01, -4.468108067381301e-01, -4.680577229566311e-01, -5.019660298240659e-01, -5.019660298240659e-01, -5.399422586873893e-02, -1.080109858560442e-01, -5.105144993684474e-02, -1.953329454726483e+00, -5.892793632551279e-02, -5.892793632551279e-02, -1.090799579331776e-03, -1.243326204037387e-03, -9.368628861923023e-04, -2.326232089865393e-02, -1.134979978243356e-03, -1.134979978243356e-03, -4.924623021582597e-01, -4.899994115608676e-01, -4.908805552751118e-01, -4.915610153092356e-01, -4.912201392846096e-01, -4.912201392846096e-01, -4.714213244691187e-01, -3.993083769153480e-01, -4.196563443988559e-01, -4.390204630922545e-01, -4.290568839306820e-01, -4.290568839306819e-01, -5.616138124142285e-01, -1.425954919925175e-01, -1.751877501062532e-01, -2.399805758454984e-01, -2.040322688850634e-01, -2.040322688850634e-01, -3.470938289161830e-01, -8.551365847233603e-03, -1.901975195199362e-02, -2.168600872612315e-01, -3.645528981939924e-02, -3.645528981939921e-02, -2.691298111714144e-03, -3.459093898546629e-04, -6.534029552770648e-04, -3.231648192137401e-02, -9.674109702629910e-04, -9.674109702629896e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lb07_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lb07", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.998307407697127e-11, 1.998283025312687e-11, 1.998156299182701e-11, 1.998555943171841e-11, 1.998294699310019e-11, 1.998294699310019e-11, 1.941035367636133e-07, 1.941014617873712e-07, 1.940037584009783e-07, 1.939626243312698e-07, 1.940982578554964e-07, 1.940982578554964e-07, 8.188294583145949e-04, 8.229612172897952e-04, 9.392202774834822e-04, 9.038726529050120e-04, 8.203335590079417e-04, 8.203335590079417e-04, 2.802907265445851e-01, 2.660992526532358e-01, 3.351879760746208e-04, 9.213369320751712e-01, 2.759425126349042e-01, 2.759425126349042e-01, 3.492011537461019e-03, 6.798784721272808e-03, 1.009575580545413e+01, 6.437008703281114e-11, 5.869760915961791e-03, 5.869760915961791e-03, 2.496506468495039e-08, 2.494252915080496e-08, 2.496268585046673e-08, 2.494516414154044e-08, 2.495360758116016e-08, 2.495360758116016e-08, 3.657220590093240e-06, 3.528686022426295e-06, 3.710763864394874e-06, 3.608217161079906e-06, 3.503594550695078e-06, 3.503594550695078e-06, 1.608553463082988e-03, 1.075852673374985e-03, 2.072553905954766e-03, 1.691003359109473e-03, 1.339634053908640e-03, 1.339634053908640e-03, 2.459110607555532e+00, 2.725471770621767e-01, 2.551814811025595e+00, 4.036360387979658e-06, 1.583168968428811e+00, 1.583168968428811e+00, 1.700547574184682e-11, 1.832844738529670e-09, 2.853557702757779e-14, 7.287824557197264e+00, 7.580605232291588e-11, 7.580605232291588e-11, 1.381448783428155e-03, 1.417832492624572e-03, 1.404660289329691e-03, 1.394597275311293e-03, 1.399623080721118e-03, 1.399623080721118e-03, 1.618271693609610e-03, 3.206152220728214e-03, 2.602549464526362e-03, 2.159914609108061e-03, 2.373220412016903e-03, 2.373220412016904e-03, 8.535042267501736e-04, 1.073631672232308e-01, 5.149085884792539e-02, 1.723392138126548e-02, 2.996562309032247e-02, 2.996562309032250e-02, 5.233207263602260e-03, 1.017762679236578e+01, 8.470785673413147e+00, 2.228582406638550e-02, 4.269436345204238e+00, 4.269436345204237e+00, 2.897827319915950e-02, 9.092583681632396e-50, 4.803201276088799e-23, 4.763637692033264e+00, 1.204469737208259e-13, 1.204469737208182e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05