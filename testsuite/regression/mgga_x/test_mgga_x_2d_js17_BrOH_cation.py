
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_2d_js17_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.506729218921575e+02, -1.506735996654439e+02, -1.506767404365257e+02, -1.506666539976960e+02, -1.506719209408269e+02, -1.506719209408269e+02, -9.643271707571722e+00, -9.643310955382523e+00, -9.644605662657611e+00, -9.645432401854876e+00, -9.643821784303139e+00, -9.643821784303139e+00, -9.298197211812911e-01, -9.305163114029442e-01, -9.554579932515399e-01, -9.639411435723627e-01, -9.702648087260681e-01, -9.702648087260681e-01, -3.513539889681574e-01, -3.437883892524991e-01, -1.105547938080255e+00, -4.110300898760533e-01, -3.747836411317407e-01, -3.747836411317407e-01, -9.536876462619212e-01, -9.234315729860204e-01, -6.465576041992196e-01, -1.154469730447699e+00, -9.779580983971633e-01, -9.779580983971625e-01, -1.788882313110859e+01, -1.789581646902340e+01, -1.788915547352380e+01, -1.789532876487326e+01, -1.789236052391215e+01, -1.789236052391215e+01, -4.293177574006076e+00, -4.328757097954556e+00, -4.282612423599005e+00, -4.313555146666326e+00, -4.318522863150115e+00, -4.318522863150115e+00, -7.139620557820145e-01, -7.555340943698516e-01, -6.501723941062940e-01, -6.311942198621048e-01, -7.234401010511400e-01, -7.234401010511400e-01, -4.969293327464875e-01, -4.561680148477828e-01, -4.993218667377909e-01, -3.985796847636080e+00, -4.227410045252209e-01, -4.227410045252209e-01, -1.134805713476293e+00, -1.101761758835992e+00, -6.907417439004082e-01, -5.222211689346019e-01, -8.121866377398260e-01, -8.121866377398266e-01, -6.785746326903450e-01, -6.689224904781415e-01, -6.704021504417972e-01, -6.728931840332828e-01, -6.714676476322334e-01, -6.714676476322334e-01, -6.571748863968592e-01, -6.267561168475668e-01, -6.160479747334031e-01, -6.127904550782259e-01, -6.126908500238371e-01, -6.126908500238371e-01, -8.105878756661218e-01, -4.614685284402799e-01, -4.450339251880308e-01, -4.207245108777714e-01, -4.245251303538838e-01, -4.245251303538836e-01, -5.660900124393020e-01, -6.762426698938578e-01, -6.076565500195256e-01, -3.669506291927068e-01, -4.550199767489637e-01, -4.550199767489636e-01, -9.096126282078348e-01, -9.786051611724478e-01, -1.017012432781722e+00, -4.672586861325591e-01, -7.765852294345625e-01, -7.765852294345624e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_2d_js17_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.303539201586545e+02, -2.303534715339172e+02, -2.303550213648278e+02, -2.303542715313189e+02, -2.303587824334289e+02, -2.303595942017845e+02, -2.303463400377847e+02, -2.303437372184721e+02, -2.303545649067307e+02, -2.303502508637922e+02, -2.303545649067307e+02, -2.303502508637922e+02, -1.451672266993462e+01, -1.451750506489439e+01, -1.451692934335384e+01, -1.451773752276556e+01, -1.452220313469858e+01, -1.452371819767436e+01, -1.451669408380570e+01, -1.451821183652545e+01, -1.451251310600972e+01, -1.452331226435288e+01, -1.451251310600972e+01, -1.452331226435288e+01, -8.997831678385328e-01, -9.194172154565211e-01, -8.927810018737434e-01, -9.168118910164371e-01, -8.014337764859660e-01, -7.694925912105575e-01, -7.963959652502464e-01, -8.069201768352569e-01, -9.518668443111453e-01, -6.317792084751402e-01, -9.518668443111453e-01, -6.317792084751402e-01, 4.111314701215842e-02, 3.140959978384022e-02, 3.601344330931252e-02, 2.473729157391073e-02, -1.154318239501480e+00, -1.284559118475585e+00, 9.282136470212424e-02, 9.121848401354624e-02, 3.796025764471440e-02, 1.245491149049645e-01, 3.796025764471443e-02, 1.245491149049640e-01, 3.157459041233645e-01, 3.107951032513871e-01, 3.066002500124086e-01, 3.005615489924472e-01, 2.114774796460119e-01, 2.064025493236842e-01, 3.807288169440652e-01, 3.853454065619229e-01, 3.275112150175837e-01, 2.974922423672304e-01, 3.275112150175831e-01, 2.974922423672293e-01, -2.758057480011853e+01, -2.757041814330794e+01, -2.759188743197727e+01, -2.758134827277179e+01, -2.758122987193689e+01, -2.757081264686861e+01, -2.759093704542716e+01, -2.758074235002458e+01, -2.758635332849304e+01, -2.757590834245231e+01, -2.758635332849304e+01, -2.757590834245231e+01, -5.258636747485523e+00, -5.258148223573932e+00, -5.343642658069326e+00, -5.340752284925053e+00, -5.150844940673383e+00, -5.181710891360892e+00, -5.224388607635178e+00, -5.255793223019229e+00, -5.386013516177655e+00, -5.312647915373574e+00, -5.386013516177655e+00, -5.312647915373574e+00, -8.410854578266453e-01, -8.377889381723157e-01, -1.120577747222537e+00, -1.123704624963886e+00, -6.871183559209207e-01, -7.453199541879107e-01, -8.789615078842959e-01, -9.235424145236693e-01, -9.197605041554409e-01, -8.380371514203337e-01, -9.197605041554409e-01, -8.380371514203337e-01, 1.405891801614994e-01, 1.385935189628525e-01, 7.817506722188189e-02, 7.742949252121091e-02, 1.458903433123473e-01, 1.411395061511453e-01, -6.164551237002220e+00, -6.160537999608759e+00, 1.171927861224454e-01, 9.852520911807128e-02, 1.171927861224454e-01, 9.852520911807128e-02, 3.848102152165667e-01, 3.961310807622682e-01, 3.670344368271476e-01, 3.749056771866638e-01, 2.356163704697457e-01, 2.243899587783583e-01, 1.611925687054787e-01, 1.617921082508334e-01, 3.010158364599845e-01, 2.621215772725770e-01, 3.010158364599849e-01, 2.621215772725764e-01, -1.041400844679318e+00, -1.048315693067451e+00, -9.834875846038926e-01, -9.909451918772315e-01, -1.003334895452371e+00, -1.010811681100867e+00, -1.020580561279800e+00, -1.027666007383850e+00, -1.011906822977541e+00, -1.019188978137062e+00, -1.011906822977541e+00, -1.019188978137062e+00, -1.010559309927034e+00, -1.015636354326027e+00, -5.243675535325875e-01, -5.308973840414956e-01, -6.408964367158260e-01, -6.485963304420953e-01, -7.729432621595260e-01, -7.784760046754484e-01, -7.048057226157706e-01, -7.103256917345249e-01, -7.048057226157706e-01, -7.103256917345249e-01, -1.197170813339660e+00, -1.203889993451225e+00, 3.511943957102007e-02, 3.327423699153251e-02, -3.581782080287260e-02, -4.200361049750321e-02, -2.518790315232792e-01, -2.561488129904745e-01, -1.316702803826957e-01, -1.326669083695293e-01, -1.316702803826955e-01, -1.326669083695290e-01, -4.435755171375196e-01, -4.534662073197577e-01, 2.180521439370699e-01, 2.175638480011740e-01, 1.953818696250553e-01, 1.910932999001430e-01, -2.688577027134867e-01, -2.813598333545283e-01, 1.402545322127520e-01, 1.264171924445309e-01, 1.402545322127521e-01, 1.264171924445313e-01, 3.089422833248873e-01, 3.041215129767063e-01, 3.930300782178501e-01, 3.447956637011990e-01, 3.425442807833251e-01, 3.300557115414930e-01, 1.383260005440794e-01, 1.372596432212447e-01, 2.434613949028127e-01, 2.661025425172547e-01, 2.434613949028133e-01, 2.661025425172536e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.922658551223126e-08, 0.000000000000000e+00, 1.922671426034474e-08, 1.922660229244303e-08, 0.000000000000000e+00, 1.922672629200798e-08, 1.922624442394622e-08, 0.000000000000000e+00, 1.922632870773129e-08, 1.922606537831987e-08, 0.000000000000000e+00, 1.922609711721711e-08, 1.922661632479062e-08, 0.000000000000000e+00, 1.922588705013869e-08, 1.922661632479062e-08, 0.000000000000000e+00, 1.922588705013869e-08, -5.192744789194093e-07, 0.000000000000000e+00, -4.894328219003054e-07, -5.152591255438519e-07, 0.000000000000000e+00, -4.835790209478496e-07, -3.992352956778060e-07, 0.000000000000000e+00, -3.687795212301184e-07, -5.919308589983500e-07, 0.000000000000000e+00, -5.583707749560515e-07, -5.358960546963231e-07, 0.000000000000000e+00, -4.559172410036824e-07, -5.358960546963231e-07, 0.000000000000000e+00, -4.559172410036824e-07, -3.378892052441022e-02, 0.000000000000000e+00, -3.327691805173702e-02, -3.399040372582122e-02, 0.000000000000000e+00, -3.336708399802301e-02, -3.663787055512417e-02, 0.000000000000000e+00, -3.741870818758204e-02, -3.583508322863808e-02, 0.000000000000000e+00, -3.559058073168908e-02, -3.225230729638189e-02, 0.000000000000000e+00, -3.940847031397370e-02, -3.225230729638189e-02, 0.000000000000000e+00, -3.940847031397370e-02, -8.134093366611218e+00, 0.000000000000000e+00, -7.214627363502407e+00, -7.912417187961474e+00, 0.000000000000000e+00, -6.893766737730354e+00, -1.860553027258777e-02, 0.000000000000000e+00, -1.591146291915803e-02, -1.819474192672643e+01, 0.000000000000000e+00, -1.728592549365759e+01, -6.875797986152612e+00, 0.000000000000000e+00, -5.141563251937108e+01, -6.875797986152612e+00, 0.000000000000000e+00, -5.141563251937105e+01, -5.252794614231095e+05, 0.000000000000000e+00, -4.289849896890312e+05, -4.549799830940420e+05, 0.000000000000000e+00, -3.621335539933469e+05, -1.426059741834788e+03, 0.000000000000000e+00, -1.194038782431587e+03, -2.773122850641593e+06, 0.000000000000000e+00, -2.914017973383501e+06, -7.656704198806606e+05, 0.000000000000000e+00, -5.950628437482995e+06, -7.656704198806608e+05, 0.000000000000000e+00, -5.950628437482994e+06, 5.173397764202373e-06, 0.000000000000000e+00, 5.175722675083314e-06, 5.216833989869957e-06, 0.000000000000000e+00, 5.217735950732910e-06, 5.175585211276612e-06, 0.000000000000000e+00, 5.177005575285486e-06, 5.212854663880717e-06, 0.000000000000000e+00, 5.215171842558645e-06, 5.195792072336375e-06, 0.000000000000000e+00, 5.196829373110545e-06, 5.195792072336375e-06, 0.000000000000000e+00, 5.196829373110545e-06, -2.536410560704326e-04, 0.000000000000000e+00, -2.536917069957840e-04, -2.437466532483216e-04, 0.000000000000000e+00, -2.440799849575949e-04, -2.640995233441956e-04, 0.000000000000000e+00, -2.611464637699603e-04, -2.553376303465710e-04, 0.000000000000000e+00, -2.523443289395061e-04, -2.402811939081553e-04, 0.000000000000000e+00, -2.475292640681972e-04, -2.402811939081553e-04, 0.000000000000000e+00, -2.475292640681972e-04, -5.883143330268958e-02, 0.000000000000000e+00, -5.951974209637823e-02, -1.270489002903063e-02, 0.000000000000000e+00, -1.143001334849187e-02, -9.375483862579648e-02, 0.000000000000000e+00, -7.897382442184224e-02, -5.154369421052092e-02, 0.000000000000000e+00, -4.379833857449869e-02, -4.780826986154182e-02, 0.000000000000000e+00, -6.138459370077551e-02, -4.780826986154184e-02, 0.000000000000000e+00, -6.138459370077549e-02, -5.563523487529331e+01, 0.000000000000000e+00, -5.425250631119004e+01, -6.396214197597477e+00, 0.000000000000000e+00, -6.289850284637676e+00, -7.843683428403480e+01, 0.000000000000000e+00, -6.605006857137586e+01, 3.893103039546993e-04, 0.000000000000000e+00, 3.906801810999421e-04, -3.759957159092792e+01, 0.000000000000000e+00, -3.325197905686330e+01, -3.759957159092792e+01, 0.000000000000000e+00, -3.325197905686330e+01, -7.722636358690266e+06, 0.000000000000000e+00, -6.602207580785557e+06, -3.279726183624926e+06, 0.000000000000000e+00, -3.073318609241894e+06, -1.209032337475252e+07, 0.000000000000000e+00, -1.022272474570584e+07, -2.810849004143406e+02, 0.000000000000000e+00, -2.728764557300819e+02, -9.930219102373650e+06, 0.000000000000000e+00, -3.423454716672709e+06, -9.930219102373645e+06, 0.000000000000000e+00, -3.423454716672708e+06, 6.318397590630896e-02, 0.000000000000000e+00, 6.539995156231150e-02, -1.936553974880137e-02, 0.000000000000000e+00, -1.765607840698605e-02, -2.715631737635392e-03, 0.000000000000000e+00, -7.685498364200795e-04, 1.860918652631343e-02, 0.000000000000000e+00, 2.036184860268047e-02, 6.817727846998388e-03, 0.000000000000000e+00, 8.684784136290752e-03, 6.817727846998388e-03, 0.000000000000000e+00, 8.684784136290752e-03, 1.560077675164730e-01, 0.000000000000000e+00, 1.551398468628564e-01, -1.314934505955277e-01, 0.000000000000000e+00, -1.291656147950018e-01, -1.090370076673921e-01, 0.000000000000000e+00, -1.068401671162093e-01, -8.237930412268334e-02, 0.000000000000000e+00, -8.080827567871329e-02, -9.707147103569176e-02, 0.000000000000000e+00, -9.533443964790210e-02, -9.707147103569176e-02, 0.000000000000000e+00, -9.533443964790210e-02, -1.214079808021623e-02, 0.000000000000000e+00, -1.064295046375350e-02, -2.830721583933649e+00, 0.000000000000000e+00, -2.777570827136415e+00, -1.541998585707804e+00, 0.000000000000000e+00, -1.494826541551584e+00, -5.927829185410433e-01, 0.000000000000000e+00, -5.799867985209609e-01, -9.589250017330918e-01, 0.000000000000000e+00, -9.624099831390187e-01, -9.589250017330921e-01, 0.000000000000000e+00, -9.624099831390188e-01, -1.900342366501418e-01, 0.000000000000000e+00, -1.848656190693506e-01, -1.487281111475522e+03, 0.000000000000000e+00, -1.455083453099644e+03, -5.637370950824360e+02, 0.000000000000000e+00, -5.086852661031710e+02, -7.465604590624665e-01, 0.000000000000000e+00, -7.015676996730223e-01, -1.404430277662847e+02, 0.000000000000000e+00, -1.228043464837714e+02, -1.404430277662847e+02, 0.000000000000000e+00, -1.228043464837714e+02, -1.547175961039527e+05, 0.000000000000000e+00, -1.381999954672732e+05, -3.277336502898533e+08, 0.000000000000000e+00, -3.695087600914025e+08, -2.794712421819000e+07, 0.000000000000000e+00, -2.309166047040601e+07, -1.649256020171145e+02, 0.000000000000000e+00, -1.556952925767834e+02, -1.367846486475834e+07, 0.000000000000000e+00, -4.597773159299823e+06, -1.367846486475838e+07, 0.000000000000000e+00, -4.597773159299842e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.553263701245229e-05, 3.553270820118372e-05, 3.553253083993188e-05, 3.553263103486755e-05, 3.553207760117776e-05, 3.553201337536334e-05, 3.553327953482676e-05, 3.553354442871649e-05, 3.553257942413988e-05, 3.553284886954278e-05, 3.553257942413988e-05, 3.553284886954278e-05, 4.380995564909504e-04, 4.381924828725252e-04, 4.381083281786041e-04, 4.382078103514662e-04, 4.383875540461783e-04, 4.384574438578120e-04, 4.378085162890700e-04, 4.378933811316254e-04, 4.381651061501918e-04, 4.381272341015103e-04, 4.381651061501918e-04, 4.381272341015103e-04, 3.475093699383759e-03, 3.496757840048069e-03, 3.468795751260699e-03, 3.495081696872056e-03, 3.395436806066907e-03, 3.366714321724747e-03, 3.350439274137479e-03, 3.360405103927150e-03, 3.523020574169496e-03, 3.208484020307417e-03, 3.523020574169496e-03, 3.208484020307417e-03, 1.310105130299360e-02, 1.295440927683902e-02, 1.317202581312386e-02, 1.298844338772998e-02, 2.922062314014100e-03, 2.927563896154626e-03, 1.383184152702054e-02, 1.370905112232869e-02, 1.256275568638017e-02, 1.541998343967562e-02, 1.256275568638017e-02, 1.541998343967562e-02, 3.735534743339011e-02, 3.660935197404189e-02, 3.722430716729640e-02, 3.642922105065850e-02, 2.001653569391594e-02, 1.974285279045624e-02, 4.274563789591525e-02, 4.272548757881717e-02, 3.864011072547127e-02, 5.511610963315690e-02, 3.864011072547129e-02, 5.511610963315693e-02, 2.725387942672447e-04, 2.726213045733969e-04, 2.726883134953354e-04, 2.727658867313449e-04, 2.725457345161734e-04, 2.726252963308827e-04, 2.726740403336859e-04, 2.727566506380097e-04, 2.726163927685209e-04, 2.726940628774798e-04, 2.726163927685209e-04, 2.726940628774798e-04, 8.042564643427781e-04, 8.042997086546577e-04, 8.002006486669833e-04, 8.003992112022716e-04, 8.011490881879724e-04, 8.021582611385508e-04, 7.976735191970900e-04, 7.985553733287574e-04, 8.042133075900754e-04, 8.025946111791659e-04, 8.042133075900754e-04, 8.025946111791659e-04, 4.739967490750068e-03, 4.759044392018857e-03, 5.373701676466446e-03, 5.395550360048622e-03, 5.192815284185383e-03, 5.046532612704382e-03, 6.163199148051303e-03, 5.872672833481500e-03, 4.625273176253300e-03, 4.900896260151650e-03, 4.625273176253301e-03, 4.900896260151652e-03, 1.468542524261041e-02, 1.472145124424323e-02, 1.110570235850815e-02, 1.107757747510367e-02, 1.536028788885006e-02, 1.511898631119079e-02, 1.202091166021897e-03, 1.203049705415124e-03, 1.475409293873635e-02, 1.539832166066417e-02, 1.475409293873635e-02, 1.539832166066417e-02, 4.986279322950664e-02, 4.788712149589399e-02, 4.491460766890625e-02, 4.400439601618922e-02, 6.940629093194295e-02, 6.969473010337085e-02, 1.810145011442858e-02, 1.792974344687973e-02, 5.997071151751641e-02, 5.434004107004659e-02, 5.997071151751637e-02, 5.434004107004657e-02, 6.946988328136190e-03, 6.933374230274058e-03, 6.078362991922568e-03, 6.065831843483485e-03, 6.307072504396933e-03, 6.295854606460622e-03, 6.549571344796712e-03, 6.534396176643413e-03, 6.421213212798781e-03, 6.408112790894385e-03, 6.421213212798781e-03, 6.408112790894385e-03, 7.763584820046511e-03, 7.732570314612917e-03, 5.231284202823251e-03, 5.217469409419991e-03, 5.389835250001898e-03, 5.379212232008084e-03, 5.725867788691343e-03, 5.709273527622248e-03, 5.533039435403245e-03, 5.515215068638718e-03, 5.533039435403245e-03, 5.515215068638718e-03, 4.968200778608617e-03, 4.990466373000128e-03, 9.621569951809973e-03, 9.597089971631806e-03, 8.797438391986327e-03, 8.775897414179990e-03, 7.888791122689030e-03, 7.854144518132317e-03, 8.337866457638166e-03, 8.359418244236005e-03, 8.337866457638168e-03, 8.359418244236009e-03, 5.784441789786080e-03, 5.765807787991457e-03, 1.975494441396800e-02, 1.972243713033592e-02, 1.818132617983024e-02, 1.810866838875147e-02, 8.945483576290457e-03, 8.858552671869571e-03, 1.727834755596599e-02, 1.774918342673665e-02, 1.727834755596600e-02, 1.774918342673665e-02, 3.203988216823882e-02, 3.183306945997476e-02, 8.589887900614138e-02, 9.623974517339616e-02, 6.302896224178810e-02, 6.265409729261720e-02, 1.786285464398405e-02, 1.772148074237600e-02, 7.044863528023976e-02, 5.619518736311527e-02, 7.044863528023976e-02, 5.619518736311533e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05