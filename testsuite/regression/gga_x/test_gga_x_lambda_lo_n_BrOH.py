
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lambda_lo_n_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_lo_n", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.094538977477690e+01, -2.094541363154748e+01, -2.094559652624387e+01, -2.094520434133734e+01, -2.094540171685903e+01, -2.094540171685903e+01, -3.473460662876329e+00, -3.473431298426606e+00, -3.472767778051822e+00, -3.474648000577984e+00, -3.473460356313282e+00, -3.473460356313282e+00, -6.982703330151558e-01, -6.983342246034276e-01, -7.020586533122092e-01, -7.063078874183994e-01, -6.982897422749518e-01, -6.982897422749518e-01, -2.171634218461500e-01, -2.180051359873830e-01, -8.126677427409646e-01, -1.803399434352282e-01, -2.173913988593892e-01, -2.173913988593892e-01, -1.700787290204166e-02, -1.781048197756913e-02, -7.043441988914130e-02, -8.175981115333884e-03, -1.762497252435427e-02, -1.762497252435427e-02, -5.032904267831589e+00, -5.032305029972544e+00, -5.032849294998267e+00, -5.032383242907707e+00, -5.032587835116723e+00, -5.032587835116723e+00, -2.114205151831996e+00, -2.123690439623671e+00, -2.115807096845306e+00, -2.123182065825743e+00, -2.118573586001694e+00, -2.118573586001694e+00, -5.739449604756368e-01, -5.954493036827437e-01, -5.467571770090973e-01, -5.476463906012203e-01, -5.915672444148793e-01, -5.915672444148793e-01, -1.448925925371300e-01, -2.378334241906899e-01, -1.422989949257285e-01, -1.811388264486426e+00, -1.598914203304805e-01, -1.598914203304805e-01, -7.887423677817272e-03, -9.012729623735746e-03, -6.755293436195974e-03, -9.542920024387892e-02, -8.211948622905082e-03, -8.211948622905082e-03, -5.586273415509476e-01, -5.614469295557988e-01, -5.604500277187026e-01, -5.596664535431372e-01, -5.600588781352789e-01, -5.600588781352789e-01, -5.397590081506626e-01, -5.167414268432413e-01, -5.233615274747786e-01, -5.292734165435595e-01, -5.260876844158774e-01, -5.260876844158774e-01, -6.253181715920225e-01, -2.823982255984757e-01, -3.155959916844032e-01, -3.667915555432324e-01, -3.391686515540480e-01, -3.391686515540480e-01, -4.690485174626595e-01, -6.565611315323992e-02, -8.711597678747812e-02, -3.365321049388584e-01, -1.195599173158715e-01, -1.195599173158715e-01, -1.992342295353234e-02, -2.468901940426747e-03, -4.689346881217620e-03, -1.139644060020528e-01, -6.979621814705429e-03, -6.979621814705420e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lambda_lo_n_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_lo_n", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.514204632565211e+01, -2.514213318701121e+01, -2.514252828387795e+01, -2.514110575774962e+01, -2.514209206993413e+01, -2.514209206993413e+01, -4.036660722792851e+00, -4.036699506815911e+00, -4.037932278956459e+00, -4.036664472767870e+00, -4.036697866035236e+00, -4.036697866035236e+00, -7.525879386130018e-01, -7.514447519504690e-01, -7.236748309658155e-01, -7.296681060945263e-01, -7.521720830838124e-01, -7.521720830838124e-01, -2.067537953462761e-01, -2.084697743698269e-01, -9.161904713332037e-01, -1.765580122016793e-01, -2.072387586117118e-01, -2.072387586117118e-01, -2.256235356352373e-02, -2.361526406752489e-02, -8.853704074509822e-02, -1.089059046471517e-02, -2.336954093227886e-02, -2.336954093227886e-02, -6.197133191059199e+00, -6.199812434425211e+00, -6.197407026706900e+00, -6.199490284604685e+00, -6.198507094507694e+00, -6.198507094507694e+00, -2.183408861496393e+00, -2.200124746899995e+00, -2.175494564120644e+00, -2.188420127758319e+00, -2.205398416300355e+00, -2.205398416300355e+00, -6.785992852168161e-01, -7.667001154420074e-01, -6.420316734480143e-01, -6.925308500498342e-01, -7.087644382841790e-01, -7.087644382841790e-01, -1.574764761828679e-01, -2.257712177986075e-01, -1.534931625687541e-01, -2.331754530622565e+00, -1.633304103937555e-01, -1.633304103937555e-01, -1.050665413620803e-02, -1.200216932064585e-02, -8.996512382772074e-03, -1.138233916078759e-01, -1.093565559465744e-02, -1.093565559465744e-02, -7.371364360140947e-01, -7.258685123754007e-01, -7.298438443561232e-01, -7.329643978696788e-01, -7.313983579850351e-01, -7.313983579850351e-01, -7.147047849504209e-01, -5.731735813720524e-01, -6.107147595264976e-01, -6.487765977902638e-01, -6.291947013696404e-01, -6.291947013696405e-01, -8.026866068589082e-01, -2.667173278990806e-01, -3.066010579070232e-01, -3.962983295571607e-01, -3.457623501976659e-01, -3.457623501976658e-01, -5.151866442463423e-01, -8.355229622836471e-02, -1.067438161992377e-01, -3.804527682369581e-01, -1.332645647286792e-01, -1.332645647286792e-01, -2.640454254594819e-02, -3.291470324691940e-03, -6.249925769522922e-03, -1.268552913182007e-01, -9.296065145380379e-03, -9.296065145380367e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lambda_lo_n_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lambda_lo_n", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.052021566882466e-09, -7.051978786940832e-09, -7.051700368390835e-09, -7.052402650595993e-09, -7.051999738205950e-09, -7.051999738205950e-09, -9.516717821737140e-06, -9.516934999321299e-06, -9.521330990298821e-06, -9.505687810460171e-06, -9.516667362326376e-06, -9.516667362326376e-06, -6.069135143249418e-03, -6.071338718901732e-03, -6.039267521797936e-03, -5.891574954525172e-03, -6.070059404823435e-03, -6.070059404823435e-03, -6.445695040982042e-01, -6.397381188987701e-01, -3.232912508596155e-03, -9.606095906634584e-01, -6.435743649776939e-01, -6.435743649776939e-01, -5.126022705228495e+00, -5.139807703922331e+00, -2.141975479279511e+00, -3.630959884428904e+00, -5.339404148273145e+00, -5.339404148273145e+00, -2.079765206968590e-06, -2.079974057108411e-06, -2.079777964117741e-06, -2.079940511849999e-06, -2.079884884790267e-06, -2.079884884790267e-06, -7.339393490055905e-05, -7.202475430511250e-05, -7.325966337889912e-05, -7.219983704577653e-05, -7.261335691486111e-05, -7.261335691486111e-05, -1.263014249265102e-02, -1.028585738141427e-02, -1.540225489818010e-02, -1.456279589327551e-02, -1.109637526577860e-02, -1.109637526577860e-02, -1.141903967754821e+00, -3.856954474530396e-01, -1.305692199082890e+00, -1.201303955654945e-04, -1.204574095531694e+00, -1.204574095531694e+00, -3.851903254564408e+00, -3.859829631349620e+00, -1.104761366828685e+01, -2.081777621214451e+00, -5.701160164785965e+00, -5.701160164785965e+00, -1.304028492356495e-02, -1.297471407192094e-02, -1.299813692425660e-02, -1.301650104306482e-02, -1.300735682170149e-02, -1.300735682170149e-02, -1.492285215047410e-02, -1.995192898495267e-02, -1.841800597338618e-02, -1.705679926929659e-02, -1.776090970464856e-02, -1.776090970464856e-02, -8.475848625826165e-03, -2.186356177909076e-01, -1.478033552418172e-01, -7.962581610594353e-02, -1.111667929159165e-01, -1.111667929159166e-01, -2.953965960265721e-02, -1.813799903656958e+00, -1.801766568182123e+00, -1.097654961957683e-01, -1.964242374122576e+00, -1.964242374122578e+00, -3.851019011645412e+00, -6.639114696292926e+00, -5.716436945514092e+00, -2.410633931013713e+00, -8.326105769337596e+00, -8.326105769337584e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05