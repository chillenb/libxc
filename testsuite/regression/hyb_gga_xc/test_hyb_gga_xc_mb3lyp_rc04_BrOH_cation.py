
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mb3lyp_rc04_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mb3lyp_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.684564623989234e+01, -1.684567003457027e+01, -1.684582700970148e+01, -1.684547193768929e+01, -1.684565071116543e+01, -1.684565071116543e+01, -2.861256816149603e+00, -2.861230512268514e+00, -2.860690560194137e+00, -2.862157587712990e+00, -2.861292513288370e+00, -2.861292513288370e+00, -6.148293832042446e-01, -6.146722658784435e-01, -6.118286559786257e-01, -6.159823121669736e-01, -6.151261603188692e-01, -6.151261603188692e-01, -1.897725370600191e-01, -1.915096819134263e-01, -7.053585010407929e-01, -1.556594432135189e-01, -1.795917728236893e-01, -1.795917728236892e-01, -4.889426225249768e-02, -4.899768393816272e-02, -8.871934035260422e-02, -4.417022736975389e-02, -4.474234443734097e-02, -4.474234443734094e-02, -4.122032495919134e+00, -4.121616950150213e+00, -4.122021638695185e+00, -4.121654668221577e+00, -4.121817371341773e+00, -4.121817371341773e+00, -1.751271207642677e+00, -1.759772564724310e+00, -1.750888208269586e+00, -1.758396091097245e+00, -1.756164698587436e+00, -1.756164698587436e+00, -5.244564855838546e-01, -5.475615890158844e-01, -4.908197513331958e-01, -4.920796996730320e-01, -5.304323354928129e-01, -5.304323354928131e-01, -1.219915786841776e-01, -1.965528236912889e-01, -1.166093684300962e-01, -1.536991143051043e+00, -1.338872871509972e-01, -1.338872871509972e-01, -3.974428056452772e-02, -4.256839160543444e-02, -2.797253261376500e-02, -9.746772353974863e-02, -3.398342316245638e-02, -3.398342316245640e-02, -5.044837328183475e-01, -5.064836771008854e-01, -5.058259555628091e-01, -5.052423810970122e-01, -5.055379006315391e-01, -5.055379006315391e-01, -4.904479783533658e-01, -4.594904397848261e-01, -4.692966844086428e-01, -4.781082051534305e-01, -4.735855416673230e-01, -4.735855416673230e-01, -5.728100192956118e-01, -2.382783205515848e-01, -2.743353572339221e-01, -3.347923837787621e-01, -3.027739999459113e-01, -3.027739999459114e-01, -4.259785522031245e-01, -8.890288331595030e-02, -9.438655406887658e-02, -3.178511142742293e-01, -1.058473479506567e-01, -1.058473479506567e-01, -5.553578641299955e-02, -2.433921798544392e-02, -3.202174138935260e-02, -1.029318890573174e-01, -3.194655699864852e-02, -3.194655699864851e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mb3lyp_rc04_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mb3lyp_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.036995404736036e+01, -2.036992985555058e+01, -2.037003967189331e+01, -2.036999235153580e+01, -2.037030076891452e+01, -2.037036839295880e+01, -2.036933016926866e+01, -2.036913571079172e+01, -2.037000295327337e+01, -2.036962042898580e+01, -2.037000295327337e+01, -2.036962042898580e+01, -3.366481473693893e+00, -3.366553462275821e+00, -3.366508843803987e+00, -3.366582201739916e+00, -3.367188887763885e+00, -3.367360621723506e+00, -3.366545391856755e+00, -3.366712122768942e+00, -3.365942396155244e+00, -3.367317281780843e+00, -3.365942396155244e+00, -3.367317281780843e+00, -7.030563747384091e-01, -7.047984406318939e-01, -7.020192479265362e-01, -7.041347229091117e-01, -6.843094251821688e-01, -6.821762443919245e-01, -6.881050704329194e-01, -6.887857005119575e-01, -7.000409364424743e-01, -6.798127049521236e-01, -7.000409364424743e-01, -6.798127049521236e-01, -2.117122996786037e-01, -2.120623786942429e-01, -2.136042251092038e-01, -2.141956665631819e-01, -8.024611126976460e-01, -8.208993573889044e-01, -1.749112292972447e-01, -1.752551445435734e-01, -1.803881554703648e-01, -1.921520935093602e-01, -1.803881554703647e-01, -1.921520935093602e-01, -1.576388587107839e-02, -1.580187752158400e-02, -1.616903165608924e-02, -1.619536524144748e-02, -5.166604254785895e-02, -5.210842068118254e-02, -1.173615380789492e-02, -1.175983188005154e-02, -1.327773249478208e-02, -1.157915899077054e-02, -1.327773249478206e-02, -1.157915899077053e-02, -5.082952140863921e+00, -5.081776869948223e+00, -5.084872903198796e+00, -5.083636206339603e+00, -5.083058209805366e+00, -5.081842388286042e+00, -5.084709149069720e+00, -5.083528540095487e+00, -5.083934743684231e+00, -5.082711096583372e+00, -5.083934743684231e+00, -5.082711096583372e+00, -1.885119365692532e+00, -1.885042381585973e+00, -1.899427859716975e+00, -1.899004112302772e+00, -1.871665345092177e+00, -1.874476537882374e+00, -1.884004509772766e+00, -1.886974977312994e+00, -1.902528736362814e+00, -1.895058077023020e+00, -1.902528736362814e+00, -1.895058077023020e+00, -6.314748707775394e-01, -6.304969635997144e-01, -6.947117142667136e-01, -6.950980360498621e-01, -5.799885715458910e-01, -5.926229729616639e-01, -6.127959848191763e-01, -6.249003792550712e-01, -6.514643924039899e-01, -6.313718486883879e-01, -6.514643924039900e-01, -6.313718486883880e-01, -1.357703715018064e-01, -1.354962004518373e-01, -2.234363405464082e-01, -2.234691317703168e-01, -1.251585648691547e-01, -1.262413794247836e-01, -1.960154569057262e+00, -1.959443308039910e+00, -1.503781689987289e-01, -1.462862417788436e-01, -1.503781689987289e-01, -1.462862417788436e-01, -9.838684455529463e-03, -1.005267949735317e-02, -1.132641675021059e-02, -1.144668919990567e-02, -8.012875291917085e-03, -7.942736069125459e-03, -8.243031863014205e-02, -8.290589373017690e-02, -9.485943303858388e-03, -9.706586096311738e-03, -9.485943303858409e-03, -9.706586096311738e-03, -6.511574059237571e-01, -6.528216814493504e-01, -6.428306214697579e-01, -6.444994632915743e-01, -6.456194625665189e-01, -6.473019726454867e-01, -6.480450046225904e-01, -6.496986003937559e-01, -6.468194086270149e-01, -6.484872518443283e-01, -6.468194086270149e-01, -6.484872518443283e-01, -6.366932826521886e-01, -6.380204320039013e-01, -5.315891780460843e-01, -5.328004628013996e-01, -5.579719473451601e-01, -5.594400402569788e-01, -5.869495292150204e-01, -5.882194350594023e-01, -5.719272778892027e-01, -5.731917373566554e-01, -5.719272778892027e-01, -5.731917373566554e-01, -7.245483961771841e-01, -7.256075445388630e-01, -2.669904032039000e-01, -2.670439881044993e-01, -3.062662456871790e-01, -3.065608843844339e-01, -3.876518869458715e-01, -3.885503268544684e-01, -3.427619753631548e-01, -3.426216703959467e-01, -3.427619753631549e-01, -3.426216703959468e-01, -4.928805248128338e-01, -4.945723164695239e-01, -5.031287024890348e-02, -5.035957131330680e-02, -6.670377617122340e-02, -6.727083262359455e-02, -3.753289498678466e-01, -3.778437390517562e-01, -1.041696660619983e-01, -1.029929997121880e-01, -1.041696660619983e-01, -1.029929997121879e-01, -1.989552775680519e-02, -1.985231533493714e-02, -5.180780036112016e-03, -4.765124937627091e-03, -7.660415522454890e-03, -7.645677784775188e-03, -9.733501928935769e-02, -9.783114921074218e-02, -8.416103095010008e-03, -9.266287350738457e-03, -8.416103095010018e-03, -9.266287350738442e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mb3lyp_rc04_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mb3lyp_rc04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.043401602249309e-08, 1.185558425401183e-11, -1.043407505857723e-08, -1.043395782851073e-08, 1.185541524986055e-11, -1.043403228730844e-08, -1.043364085111230e-08, 1.185461176244829e-11, -1.043361737800369e-08, -1.043430726173668e-08, 1.185712748825859e-11, -1.043445901963862e-08, -1.043399103588032e-08, 1.185581677785285e-11, -1.043402525603493e-08, -1.043399103588032e-08, 1.185581677785285e-11, -1.043402525603493e-08, -1.358064788891638e-05, 1.541225587214213e-07, -1.358577088276538e-05, -1.358103965629732e-05, 1.541207867069750e-07, -1.358654212396425e-05, -1.359428505024676e-05, 1.540597121148743e-07, -1.359756006350432e-05, -1.356249691843990e-05, 1.540100170374724e-07, -1.356668218543393e-05, -1.358708059286732e-05, 1.541018002171290e-07, -1.357812524846537e-05, -1.358708059286732e-05, 1.541018002171290e-07, -1.357812524846537e-05, -7.792259255394369e-03, 9.270722799357577e-04, -7.849951520308237e-03, -7.786461986509554e-03, 9.324368204854546e-04, -7.857131339723945e-03, -7.832736409582442e-03, 1.069807695322527e-03, -7.733242440070735e-03, -7.559812284896936e-03, 1.024962868820697e-03, -7.593914356376238e-03, -8.112308292553405e-03, 1.115015704194649e-03, -7.033461375403323e-03, -8.112308292553405e-03, 1.115015704194649e-03, -7.033461375403323e-03, -8.015022929288295e-01, 4.870430005817489e-01, -7.816385693227824e-01, -7.925801894797767e-01, 4.601020753985704e-01, -7.665752528935305e-01, -4.505851044012250e-03, 4.311409659161154e-04, -4.390509886285057e-03, -1.405355168870086e+00, 1.357480112752327e+00, -1.380998360113386e+00, -1.027527050921904e+00, 1.929431429665579e+00, -3.898047502256749e+00, -1.027527050921904e+00, 1.929431429665580e+00, -3.898047502256753e+00, -1.601233046657358e+04, 9.006507517295055e-08, -1.341327593453690e+04, -1.420990979929685e+04, 3.695673560289628e-07, -1.164774806483721e+04, -8.370059792973863e+01, 1.489288214502471e+01, -7.190430426370420e+01, -6.750537472164971e+04, 3.350885389426989e-17, -7.031007753052439e+04, -2.221427161418409e+04, 8.673770695270562e-13, -1.460208813509680e+05, -2.221427161418411e+04, 8.673770695270562e-13, -1.460208813509680e+05, -3.120937442842174e-06, 1.809299517292379e-08, -3.123583621634334e-06, -3.123251342613206e-06, 1.807545599629114e-08, -3.125818583373952e-06, -3.121034151413480e-06, 1.809212347410638e-08, -3.123635403204407e-06, -3.123017007873502e-06, 1.807663525628042e-08, -3.125668909910377e-06, -3.122147659704871e-06, 1.808413833346515e-08, -3.124709253674506e-06, -3.122147659704871e-06, 1.808413833346515e-08, -3.124709253674506e-06, -1.010635912347755e-04, 3.324691278596461e-06, -1.010779524700931e-04, -9.903149681292247e-05, 3.198712461362903e-06, -9.909978705291323e-05, -1.011475640512280e-04, 3.420793545345939e-06, -1.012672271456375e-04, -9.938724522915006e-05, 3.305373228471352e-06, -9.946546298030040e-05, -9.990971384051965e-05, 3.212781934568574e-06, -9.988260283181755e-05, -9.990971384051965e-05, 3.212781934568574e-06, -9.988260283181755e-05, -1.605154024547233e-02, 1.825453754003035e-03, -1.619117711055979e-02, -1.474343620688189e-02, 1.201192443406202e-03, -1.477479346054838e-02, -2.204666172622624e-02, 2.748455987594796e-03, -2.031678606014923e-02, -2.418338119192188e-02, 2.257366444272562e-03, -2.171147066780003e-02, -1.455920912893613e-02, 1.758833807333089e-03, -1.681843677980908e-02, -1.455920912893614e-02, 1.758833807333089e-03, -1.681843677980908e-02, -3.150401852206798e+00, 4.345108400965211e+00, -3.170434965885433e+00, -5.502177821396236e-01, 4.801730708975269e-01, -5.493965502748231e-01, -4.262172914865188e+00, 5.403210563745951e+00, -4.000430924985834e+00, -1.886068550772652e-04, 3.686127133635989e-06, -1.889444393377877e-04, -2.359804503894215e+00, 2.723523603638179e+00, -2.609535107084575e+00, -2.359804503894215e+00, 2.723523603638179e+00, -2.609535107084575e+00, -1.699411089643476e+05, 4.391412325733211e-24, -1.460796247556560e+05, -7.954666730295906e+04, 9.952946977518016e-18, -7.449506296332044e+04, -2.990717847119347e+05, 4.581716902253058e-25, -2.612737931159296e+05, -1.696004532565848e+01, 1.218782134268272e+01, -1.643550682122481e+01, -2.313967161586600e+05, 7.784361458058929e-20, -9.240488802955455e+04, -2.313967161586598e+05, 7.784361458058927e-20, -9.240488802955452e+04, -2.219495429854996e-02, 1.781738515145262e-03, -2.195509733342586e-02, -2.076312885154710e-02, 1.834071765417697e-03, -2.055303334584081e-02, -2.118788746189437e-02, 1.815455511527695e-03, -2.097177410252597e-02, -2.159933539387829e-02, 1.800280877020669e-03, -2.136816452351270e-02, -2.138649414498954e-02, 1.807857598825492e-03, -2.116295032564181e-02, -2.138649414498954e-02, 1.807857598825492e-03, -2.116295032564181e-02, -2.560950998573218e-02, 2.045475405717882e-03, -2.533979798454380e-02, -2.658669298607634e-02, 4.372269458584065e-03, -2.639619554799486e-02, -2.539317146300312e-02, 3.481409492639727e-03, -2.519928883854928e-02, -2.474558936546363e-02, 2.801304754186088e-03, -2.452658703155491e-02, -2.505723377368717e-02, 3.126795900343610e-03, -2.483093712094919e-02, -2.505723377368717e-02, 3.126795900343610e-03, -2.483093712094919e-02, -1.210663724077704e-02, 9.448281971276932e-04, -1.210774703956818e-02, -2.896818500974693e-01, 1.841168313207797e-01, -2.893079402659515e-01, -1.863619421801817e-01, 8.322264776862785e-02, -1.860558812379716e-01, -9.818708946814067e-02, 2.335777938713061e-02, -9.716063264790924e-02, -1.367145607178618e-01, 4.455605659310826e-02, -1.373866162956502e-01, -1.367145607178618e-01, 4.455605659310827e-02, -1.373866162956503e-01, -3.643622074116785e-02, 6.561578635696272e-03, -3.608494043291562e-02, -8.685719210089600e+01, 1.430892164753827e+01, -8.522852476782363e+01, -3.265583822645921e+01, 1.521830626896655e+01, -3.015511765247127e+01, -1.291093042631665e-01, 2.777420884472484e-02, -1.254546410894600e-01, -8.387690733914193e+00, 7.858164809404253e+00, -8.331015966778173e+00, -8.387690733914191e+00, 7.858164809404253e+00, -8.331015966778182e+00, -5.373095280702843e+03, 2.768157770177566e-04, -4.881149980613249e+03, -5.023960912792954e+06, 2.308592770889069e-85, -5.846008400586689e+06, -5.637392051558921e+05, 1.346473245321597e-36, -4.801493238580819e+05, -1.031515996738637e+01, 8.933405596013928e+00, -9.898975829389819e+00, -3.312813801870204e+05, 6.624839855865914e-22, -1.199647090775366e+05, -3.312813801870212e+05, 6.624839855865567e-22, -1.199647090775370e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05