
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_meyer_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_meyer", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.174996116528220e+03, 2.175003903450670e+03, 2.175050427609714e+03, 2.174922680130402e+03, 2.175000124575995e+03, 2.175000124575995e+03, 5.874086185703620e+01, 5.874045980015169e+01, 5.873426293331782e+01, 5.876986653301236e+01, 5.874115552209975e+01, 5.874115552209975e+01, 2.277568962898113e+00, 2.276003148253150e+00, 2.248458048235376e+00, 2.278377365124311e+00, 2.276981113024025e+00, 2.276981113024025e+00, 2.080582425767570e-01, 2.097799659351164e-01, 3.161763815097905e+00, 1.611363537553764e-01, 2.085116940335962e-01, 2.085116940335962e-01, 4.863079095486851e-01, 4.839210455904569e-01, 4.368686250739483e-01, 5.946181261920459e-01, 4.748260168349310e-01, 4.748260168349310e-01, 1.273970485108747e+02, 1.274064771037138e+02, 1.273982402954503e+02, 1.274055675678779e+02, 1.274015526580042e+02, 1.274015526580042e+02, 2.041066690166878e+01, 2.062807253253581e+01, 2.039513327708608e+01, 2.056357709568938e+01, 2.058055243613220e+01, 2.058055243613220e+01, 1.619332275798230e+00, 1.829214272512515e+00, 1.463918035800506e+00, 1.531418858467249e+00, 1.733133511512535e+00, 1.733133511512535e+00, 1.510983624203447e-01, 2.588843625705376e-01, 1.404646480173229e-01, 1.692526190618446e+01, 1.432911449720572e-01, 1.432911449720572e-01, 5.774747747625156e-01, 5.757318970706488e-01, 3.404498171398126e-01, 1.452989930511972e-01, 4.736849786136921e-01, 4.736849786136921e-01, 1.632815356482894e+00, 1.630069188078113e+00, 1.631048862998941e+00, 1.631798481727882e+00, 1.631420586859295e+00, 1.631420586859299e+00, 1.527419765753409e+00, 1.266972176868136e+00, 1.336625286172649e+00, 1.405318801990245e+00, 1.369326177310734e+00, 1.369326177310734e+00, 2.013780930401509e+00, 3.528208460764048e-01, 4.417561862114405e-01, 6.292763128731954e-01, 5.217906337297894e-01, 5.217906337297871e-01, 1.038298421153969e+00, 5.604987682806917e-01, 2.316792672198169e-01, 5.430246519976026e-01, 1.185263980038664e-01, 1.185263980038661e-01, 5.571934421343128e-01, 4.423437998746507e-01, 4.757815818419488e-01, 1.067842961619239e-01, 3.923948076383683e-01, 3.923948076383405e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_meyer_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_meyer", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.338894493914416e+03, 3.338913536281659e+03, 3.339006329657017e+03, 3.338694347965861e+03, 3.338904470669732e+03, 3.338904470669732e+03, 8.737915904106983e+01, 8.738019428210359e+01, 8.741582485112234e+01, 8.739145044705846e+01, 8.738043595571099e+01, 8.738043595571099e+01, 3.086315089990897e+00, 3.076699244315269e+00, 2.816753582269745e+00, 2.867337758453834e+00, 3.082824533103609e+00, 3.082824533103609e+00, 1.892295823638374e-01, 1.987718303348772e-01, 4.548148132802075e+00, 3.160179346091539e-02, 1.921533941706473e-01, 1.921533941706473e-01, -5.255822317878984e-01, -5.270571500940364e-01, -1.333193228055471e+00, -6.035813421992238e-01, -5.170668058867476e-01, -5.170668058867476e-01, 1.999944421348670e+02, 2.001020515877567e+02, 2.000055948912777e+02, 2.000892658588647e+02, 2.000494270582189e+02, 2.000494270582189e+02, 2.566942828233943e+01, 2.610903881661035e+01, 2.541501632904260e+01, 2.575770149664806e+01, 2.629538368952678e+01, 2.629538368952678e+01, 2.450713145800973e+00, 2.974515346695484e+00, 2.200641816578633e+00, 2.456438634588559e+00, 2.655994695789564e+00, 2.655994695789564e+00, -9.965548677366641e-02, 1.329268188194242e-01, -8.250173026895589e-02, 2.751732148810120e+01, -1.955445824269161e-02, -1.955445824269161e-02, -5.858154575034892e-01, -5.866317288683535e-01, -3.465721387951760e-01, -3.691501825039941e-01, -4.827342566828740e-01, -4.827342566829306e-01, 2.702094069014456e+00, 2.658622633314256e+00, 2.674189182960824e+00, 2.686221552352023e+00, 2.680200246689747e+00, 2.680200246689753e+00, 2.533710073831362e+00, 1.786499390548120e+00, 1.996646115897616e+00, 2.197724620513228e+00, 2.095021569484897e+00, 2.095021569484893e+00, 3.267180301047831e+00, 2.812478953382274e-01, 4.617460352607354e-01, 8.558625711939685e-01, 6.379271545174245e-01, 6.379271545174172e-01, 1.445314881706769e+00, -1.263786532718447e+00, -1.656799672852826e+00, 7.836713766613800e-01, -1.107552827378434e-01, -1.107552827378434e-01, -6.112508916085647e-01, -4.431590292127205e-01, -4.787250723322854e-01, -9.802012140401815e-02, -3.989288213292628e-01, -3.989288213292725e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_meyer_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_meyer", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.244075794917738e-07, 7.244021493796850e-07, 7.243742645696306e-07, 7.244632601697676e-07, 7.244047464281184e-07, 7.244047464281184e-07, 1.684059022850585e-04, 1.684045241149726e-04, 1.683458692087992e-04, 1.683398871938369e-04, 1.684030413066922e-04, 1.684030413066922e-04, 2.413629042680612e-02, 2.421672811162217e-02, 2.646193540670357e-02, 2.583471490067120e-02, 2.416555459932498e-02, 2.416555459932498e-02, 1.226387192199107e+00, 1.174056283725167e+00, 1.393594929487160e-02, 3.562469939970334e+00, 1.210287403558777e+00, 1.210287403558777e+00, 5.965366417934025e+04, 5.190954108065215e+04, 8.213663189174795e+02, 5.402256535180162e+05, 5.356677119131778e+04, 5.356677119131778e+04, 4.996391606507765e-05, 4.993493609468766e-05, 4.996087017608484e-05, 4.993833762202424e-05, 4.994916496853034e-05, 4.994916496853034e-05, 9.641039059210361e-04, 9.434014662813811e-04, 9.732910971583232e-04, 9.568662823647773e-04, 9.385542241317898e-04, 9.385542241317898e-04, 3.616949907462490e-02, 2.801654182626783e-02, 4.236193183545860e-02, 3.708953192412427e-02, 3.226059472634897e-02, 3.226059472634897e-02, 1.123837505187974e+01, 1.260655204150201e+00, 1.140750174980773e+01, 9.956147867389777e-04, 6.242298650858998e+00, 6.242298650858998e+00, 6.017494990986739e+05, 4.031482974380452e+05, 9.575042495406880e+05, 9.486178326100278e+01, 5.329519302034983e+05, 5.329519302035042e+05, 3.262579360650686e-02, 3.320226267650535e-02, 3.299402518622799e-02, 3.283455464161159e-02, 3.291423056263956e-02, 3.291423056264255e-02, 3.597143815434946e-02, 5.600992858228176e-02, 4.885975039683301e-02, 4.329229800985813e-02, 4.601227605395811e-02, 4.601227605395812e-02, 2.431139354171680e-02, 6.108032875946732e-01, 3.553100326762687e-01, 1.656187459475331e-01, 2.419631185400966e-01, 2.419631185400994e-01, 7.643669520117888e-02, 9.993621764202201e+02, 3.911334378773596e+02, 1.951779893056537e-01, 2.315891317301787e+01, 2.315891317301785e+01, 3.705885584966558e+04, 1.964459388737481e+07, 2.865711205598309e+06, 2.650969234280549e+01, 8.682268583994341e+05, 8.682268583993498e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05