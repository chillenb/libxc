
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_vmt_ge_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_ge", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.047928073695286e+01, -2.047931900486002e+01, -2.047953699580581e+01, -2.047896666289625e+01, -2.047925869770577e+01, -2.047925869770577e+01, -3.373292247803770e+00, -3.373271074726753e+00, -3.372864516860179e+00, -3.374212287477615e+00, -3.373348178125628e+00, -3.373348178125628e+00, -6.683867325544951e-01, -6.681749517897884e-01, -6.647086132874580e-01, -6.694873085948300e-01, -6.678825747014717e-01, -6.678825747014717e-01, -2.008088373669970e-01, -2.017951705573303e-01, -7.739366991488208e-01, -1.728049423762581e-01, -1.828048067295438e-01, -1.828048067295439e-01, -5.685254659159980e-03, -6.045924745154723e-03, -5.824146203496432e-02, -3.231379599389238e-03, -4.063219050572524e-03, -4.063219050572524e-03, -4.949411754451067e+00, -4.949402073106162e+00, -4.949419062918981e+00, -4.949410394743784e+00, -4.949402510301458e+00, -4.949402510301458e+00, -2.004791379998417e+00, -2.015415472166953e+00, -2.003190264067512e+00, -2.012536648193045e+00, -2.011472674214713e+00, -2.011472674214713e+00, -5.658370343088025e-01, -5.979573168983323e-01, -5.262387422044276e-01, -5.315185599919031e-01, -5.727342367756335e-01, -5.727342367756335e-01, -1.380359802111505e-01, -2.171432482605725e-01, -1.297955411705597e-01, -1.799771149904922e+00, -1.497675527331739e-01, -1.497675527331739e-01, -2.493218717468727e-03, -3.158963671660036e-03, -2.415954775807200e-03, -9.296898749810292e-02, -2.910429885579340e-03, -2.910429885579340e-03, -5.491876155806131e-01, -5.496441255568894e-01, -5.494866879810028e-01, -5.493496867399928e-01, -5.494178587753136e-01, -5.494178587753136e-01, -5.332537430199026e-01, -4.897283474194516e-01, -5.014575665201858e-01, -5.134243267471378e-01, -5.071272666629743e-01, -5.071272666629743e-01, -6.275336093898974e-01, -2.572079142612517e-01, -2.901121680466866e-01, -3.493797642086310e-01, -3.165621941048519e-01, -3.165621941048519e-01, -4.516301192667849e-01, -5.542613121707034e-02, -7.642396713228596e-02, -3.298284754984592e-01, -1.121872780958418e-01, -1.121872780958418e-01, -8.680057241883988e-03, -8.444081655323325e-04, -1.772450099579842e-03, -1.069607948980095e-01, -2.692920497642078e-03, -2.692920497642076e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_vmt_ge_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_ge", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.561249266278336e+01, -2.561258584309556e+01, -2.561298725394945e+01, -2.561160117807626e+01, -2.561232928149106e+01, -2.561232928149106e+01, -4.126105101414566e+00, -4.126144979463578e+00, -4.127168002607475e+00, -4.126172949705680e+00, -4.126252378280583e+00, -4.126252378280583e+00, -7.779032675114171e-01, -7.765564669708083e-01, -7.435793281930335e-01, -7.503454732731266e-01, -7.493820075098121e-01, -7.493820075098121e-01, -1.880785988479514e-01, -1.909330935418415e-01, -9.122608506403956e-01, -1.530939746954861e-01, -1.634834940071060e-01, -1.634834940071060e-01, -8.570939948852196e-03, -9.574449177017910e-03, -8.304175587616511e-02, -4.308762047900689e-03, -5.471157366018248e-03, -5.471157366018248e-03, -6.293463332590413e+00, -6.295722870769579e+00, -6.293566483974883e+00, -6.295561242012291e+00, -6.294609220741517e+00, -6.294609220741517e+00, -2.223175959117182e+00, -2.242341310462233e+00, -2.204584443579098e+00, -2.221456723762878e+00, -2.242841862150138e+00, -2.242841862150138e+00, -6.998064014236409e-01, -7.802913491136673e-01, -6.445782689936499e-01, -6.882399455929211e-01, -7.136085076730880e-01, -7.136085076730880e-01, -1.366405078165384e-01, -1.933819195224324e-01, -1.318206132402629e-01, -2.351398459152566e+00, -1.361092469051529e-01, -1.361092469051529e-01, -3.324292325252805e-03, -4.212334131127433e-03, -3.231939795333980e-03, -1.084777650311425e-01, -3.892480952545178e-03, -3.892480952545178e-03, -7.268160237350348e-01, -7.179752135873178e-01, -7.210672404263275e-01, -7.236355848997148e-01, -7.223491633899527e-01, -7.223491633899527e-01, -7.084952947262799e-01, -5.719247692503743e-01, -6.103684582277290e-01, -6.494247657739523e-01, -6.295433052271586e-01, -6.295433052271586e-01, -8.172012588739763e-01, -2.382201412820250e-01, -2.879019701012810e-01, -4.004823854637359e-01, -3.386006593750122e-01, -3.386006593750121e-01, -5.258168948526680e-01, -8.105820802364352e-02, -9.948394240473771e-02, -3.926645335384042e-01, -1.157649349699335e-01, -1.157649349699335e-01, -1.637819353749061e-02, -1.125877554043110e-03, -2.363266800182812e-03, -1.135875713260885e-01, -3.598721423618318e-03, -3.598721423618315e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_vmt_ge_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt_ge", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.286951648610208e-09, -4.286912378087700e-09, -4.286709318369982e-09, -4.287294182644823e-09, -4.286991764250526e-09, -4.286991764250526e-09, -5.947410243370279e-06, -5.947470397233735e-06, -5.948276791299856e-06, -5.942358086795941e-06, -5.946908920449109e-06, -5.946908920449109e-06, -4.040405747748802e-03, -4.050751245840344e-03, -4.280239407847309e-03, -4.152357632813644e-03, -4.188264884536040e-03, -4.188264884536040e-03, -5.907907036397527e-01, -5.757749212042055e-01, -2.221605036116186e-03, -1.048883140335010e+00, -8.715748547210364e-01, -8.715748547210364e-01, 1.463774652453222e+02, 1.971376579527015e+02, 3.229817126688658e+00, 1.869658196365055e-01, 2.222492913635026e+01, 2.222492913635045e+01, -1.236803255926297e-06, -1.236383559003288e-06, -1.236778207763680e-06, -1.236407784092886e-06, -1.236593677344096e-06, -1.236593677344096e-06, -5.212687733556484e-05, -5.088841919546126e-05, -5.264316722668215e-05, -5.153800690537520e-05, -5.118996665217007e-05, -5.118996665217007e-05, -7.434450979710086e-03, -5.661519635821180e-03, -1.002863534882109e-02, -9.137285200633278e-03, -7.033096060706993e-03, -7.033096060706993e-03, -1.597033914107224e+00, -4.364427219429813e-01, -1.798059643952190e+00, -6.890245279726245e-05, -1.650178513561697e+00, -1.650178513561697e+00, 1.281434913500170e-03, 3.245546410994496e-01, 5.073405295339172e+01, -2.461843209972089e+00, 2.209846999249341e+01, 2.209846999249317e+01, -7.847618373816251e-03, -7.922397733045426e-03, -7.896015079489406e-03, -7.874479321927585e-03, -7.885273380050463e-03, -7.885273380050463e-03, -8.794521397171764e-03, -1.397525446617854e-02, -1.223505093078758e-02, -1.073501691850329e-02, -1.148178040156598e-02, -1.148178040156598e-02, -4.676687336580943e-03, -2.208300688447366e-01, -1.303286980782884e-01, -5.487051348919526e-02, -8.651054786224995e-02, -8.651054786224997e-02, -1.937601444350857e-02, 4.653781258751433e+00, -6.351017925620489e-01, -6.673494992107742e-02, -2.956362759769565e+00, -2.956362759769563e+00, 2.269799817393471e+02, 7.278818165684435e-28, 5.498238820076980e-06, -3.017120801205677e+00, 2.148340980065226e+01, 2.148340980065220e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05