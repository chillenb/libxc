
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_kt3_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt3", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.330590914065620e+01, -2.330592065151140e+01, -2.330607854161589e+01, -2.330588795263591e+01, -2.330591432051370e+01, -2.330591432051370e+01, -3.945700630992903e+00, -3.945642274622216e+00, -3.944201400636429e+00, -3.947514159918245e+00, -3.945687238947841e+00, -3.945687238947841e+00, -8.061456162743570e-01, -8.064861299186067e-01, -8.205236692018613e-01, -8.256335372156324e-01, -8.062630716833383e-01, -8.062630716833383e-01, -2.228384388143807e-01, -2.242198322256567e-01, -9.393809193878220e-01, -1.814051264621618e-01, -2.232305743276172e-01, -2.232305743276172e-01, -2.114456813460665e-02, -2.213702540505360e-02, -7.756229663532087e-02, -1.018821285200456e-02, -2.190711066083954e-02, -2.190711066083954e-02, -5.604879810916811e+00, -5.603578081922939e+00, -5.604754937620732e+00, -5.603742584144705e+00, -5.604199529065110e+00, -5.604199529065110e+00, -2.586322409380736e+00, -2.590559362670308e+00, -2.598774016407392e+00, -2.601835933605039e+00, -2.573721809353419e+00, -2.573721809353419e+00, -6.512704037437392e-01, -6.837892033347632e-01, -6.185888922799399e-01, -6.273611784050511e-01, -6.729066281476576e-01, -6.729066281476576e-01, -1.433521823653148e-01, -2.398889262558093e-01, -1.419257059161240e-01, -2.032204377994608e+00, -1.604378414631398e-01, -1.604378414631398e-01, -9.829165570739066e-03, -1.122878933749428e-02, -8.419305698195056e-03, -9.907455106512457e-02, -1.023232325553885e-02, -1.023232325553885e-02, -6.454756877056296e-01, -6.458496359021041e-01, -6.456859751069002e-01, -6.455795778293234e-01, -6.456297545600905e-01, -6.456297545600905e-01, -6.248274510660461e-01, -5.780516324205410e-01, -5.903247869878033e-01, -6.024971491976588e-01, -5.960317700186452e-01, -5.960317700186452e-01, -7.172371297893425e-01, -2.888264845010563e-01, -3.287192092850696e-01, -3.982821524859768e-01, -3.602337303731069e-01, -3.602337303731069e-01, -5.195372527489310e-01, -7.300173312572594e-02, -9.085511766387627e-02, -3.703465265218716e-01, -1.217229521008445e-01, -1.217229521008445e-01, -2.475248273267614e-02, -3.079884404342954e-03, -5.847407737676086e-03, -1.175043179689727e-01, -8.698776667828504e-03, -8.698776667828492e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_kt3_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt3", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.641926814994702e+01, -2.641940712097107e+01, -2.641996576953715e+01, -2.641769122826819e+01, -2.641934195127945e+01, -2.641934195127945e+01, -4.166874023957077e+00, -4.166994615428843e+00, -4.170474060072184e+00, -4.165344072831875e+00, -4.166953537958553e+00, -4.166953537958553e+00, -8.166137714580894e-01, -8.142610791360871e-01, -7.490501886159253e-01, -7.556181472635434e-01, -8.157643788459999e-01, -8.157643788459999e-01, -2.328663225522502e-01, -2.365153548168332e-01, -9.961099328774310e-01, -1.933047716248744e-01, -2.339388612986667e-01, -2.339388612986667e-01, -2.796236890112948e-02, -2.925201713045508e-02, -1.051275334436755e-01, -1.356017437852486e-02, -2.894889482374736e-02, -2.894889482374736e-02, -6.669347771303884e+00, -6.674262208548156e+00, -6.669846031827804e+00, -6.673667478527150e+00, -6.671876373166663e+00, -6.671876373166663e+00, -1.806895511608071e+00, -1.839737082840273e+00, -1.772576130778911e+00, -1.798414358412574e+00, -1.873109221026121e+00, -1.873109221026121e+00, -7.933345803007746e-01, -8.873670778315771e-01, -7.546725700562987e-01, -8.090052907120576e-01, -8.262878432168445e-01, -8.262878432168445e-01, -1.869434095125821e-01, -2.456892790762097e-01, -1.797936421708381e-01, -2.587526620820761e+00, -1.836172015997806e-01, -1.836172015997806e-01, -1.308323135559573e-02, -1.493914307505340e-02, -1.120329706894488e-02, -1.362228007446265e-01, -1.361360383099916e-02, -1.361360383099916e-02, -8.496204329653066e-01, -8.419690405492869e-01, -8.448354501523191e-01, -8.469565609140993e-01, -8.459058665042291e-01, -8.459058665042291e-01, -8.238004170390769e-01, -6.727376600992756e-01, -7.213175088852790e-01, -7.639961141240410e-01, -7.428702778471404e-01, -7.428702778471404e-01, -9.269548896203952e-01, -2.936109205618998e-01, -3.504922002395866e-01, -4.784273422618512e-01, -4.093264794094796e-01, -4.093264794094796e-01, -6.110126020783607e-01, -9.930110058150128e-02, -1.288288655937537e-01, -4.648404891205720e-01, -1.559343628935590e-01, -1.559343628935589e-01, -3.268192362875965e-02, -4.105377800721661e-03, -7.790428352417150e-03, -1.465408377368611e-01, -1.157655878688023e-02, -1.157655878688021e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_kt3_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt3", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.174149360933168e-08, -1.174134139071571e-08, -1.174063374105838e-08, -1.174312703258459e-08, -1.174141357320831e-08, -1.174141357320831e-08, -1.727812509645815e-05, -1.727760386893088e-05, -1.726043793976377e-05, -1.727540437454693e-05, -1.727756355836889e-05, -1.727756355836889e-05, -9.731468958323920e-03, -9.778060966072387e-03, -1.099752258601984e-02, -1.072785251333478e-02, -9.748382585703498e-03, -9.748382585703498e-03, -4.941129318614109e-01, -4.801834080917357e-01, -5.202851601441548e-03, -6.710603678048120e-01, -4.902396603740861e-01, -4.902396603740861e-01, -9.497676451303812e+00, -9.518972699966687e+00, 4.811711104187403e+00, -6.743226610912829e+00, -9.888911333131452e+00, -9.888911333131452e+00, -3.189501923617751e-06, -3.184428768744644e-06, -3.188982196490522e-06, -3.185037530750342e-06, -3.186901159048635e-06, -3.186901159048635e-06, -1.888440073874717e-04, -1.833805647554535e-04, -1.913767999430269e-04, -1.870447270284908e-04, -1.819126640540536e-04, -1.819126640540536e-04, -1.223420042719781e-02, -7.861516587883602e-03, -1.410167803416816e-02, -1.028815984097381e-02, -1.079652240329834e-02, -1.079652240329834e-02, 1.731679596013617e-01, -3.134694955220831e-01, -1.189933481602435e-02, -1.565708164425760e-04, -5.817605518454255e-01, -5.817605518454255e-01, -7.152414615992550e+00, -7.166517168693856e+00, -2.047512570432692e+01, 2.570836742400388e+00, -1.057572856419845e+01, -1.057572856419860e+01, -7.980227995716953e-03, -8.780626100126390e-03, -8.497597180550837e-03, -8.275803567764990e-03, -8.387027098322081e-03, -8.387027098322081e-03, -8.334843865320166e-03, -1.979410156293304e-02, -1.597068970848411e-02, -1.272718841891913e-02, -1.431381288409244e-02, -1.431381288409244e-02, -7.071285825804412e-03, -1.888291886689204e-01, -1.204379340230401e-01, -4.999520012822072e-02, -8.006503750773931e-02, -8.006503750773937e-02, -2.571408291734127e-02, 5.465402764022828e+00, 4.056445584276349e+00, -4.919136559634919e-02, 2.910416429419156e-01, 2.910416429419063e-01, -7.115555094039506e+00, -1.231617761775416e+01, -1.060647084426675e+01, -4.635496251058526e-02, -1.543650559271714e+01, -1.543650559271953e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05