
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_mpwb1k_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.187648744664804e+01, -1.187650065551293e+01, -1.187660252533688e+01, -1.187638532718654e+01, -1.187649405413253e+01, -1.187649405413253e+01, -2.019162534450088e+00, -2.019145393182801e+00, -2.018742514360132e+00, -2.019650875088927e+00, -2.019163330838073e+00, -2.019163330838073e+00, -4.300989193799179e-01, -4.299294654297102e-01, -4.266283135206592e-01, -4.287121903529305e-01, -4.300345755805025e-01, -4.300345755805025e-01, -1.340931203410785e-01, -1.349590546830537e-01, -5.183191480335079e-01, -1.107088948739731e-01, -1.343438518010944e-01, -1.343438518010944e-01, -2.773062223714937e-03, -3.139001445031182e-03, -4.504234132217801e-02, -3.202589976434694e-04, -3.101459780074338e-03, -3.101459780074338e-03, -2.891801264670780e+00, -2.891450665855937e+00, -2.891768242175487e+00, -2.891495618577803e+00, -2.891617400867686e+00, -2.891617400867686e+00, -1.226461571763040e+00, -1.232015306987956e+00, -1.226819470169245e+00, -1.231135286189909e+00, -1.229782362688514e+00, -1.229782362688514e+00, -3.564665717279338e-01, -3.754019291766875e-01, -3.422200945346374e-01, -3.497157812856794e-01, -3.666486643698081e-01, -3.666486643698081e-01, -9.182751684713519e-02, -1.443017070802120e-01, -8.993142294544500e-02, -1.084239459386161e+00, -9.902092097102488e-02, -9.902092097102488e-02, -2.975751524066563e-04, -4.308490659561726e-04, -3.087494807887845e-04, -6.308545385547133e-02, -3.957175040216921e-04, -3.957175040216923e-04, -3.471346289016421e-01, -3.484900376743026e-01, -3.480554789499856e-01, -3.476877450263786e-01, -3.478761833904784e-01, -3.478761833904784e-01, -3.373442398261817e-01, -3.170578761262614e-01, -3.232427026595409e-01, -3.287300073567222e-01, -3.259198979165014e-01, -3.259198979165015e-01, -3.948029751204982e-01, -1.713778564671442e-01, -1.932955793959254e-01, -2.302950230644402e-01, -2.101339111782159e-01, -2.101339111782159e-01, -2.902181267721676e-01, -4.017824416292666e-02, -5.750452645183483e-02, -2.151112163686082e-01, -7.676037456725969e-02, -7.676037456725968e-02, -3.750749641174790e-03, -1.533330714739248e-05, -8.407724600352011e-05, -7.316723234502760e-02, -2.983068576344144e-04, -2.983068576344128e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_mpwb1k_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.423465484798893e+01, -1.423470048829428e+01, -1.423491220804961e+01, -1.423416461554340e+01, -1.423467885013786e+01, -1.423467885013786e+01, -2.377492163196262e+00, -2.377503568533665e+00, -2.377886731775277e+00, -2.377319605897687e+00, -2.377510151458804e+00, -2.377510151458804e+00, -5.037367076434822e-01, -5.029214365066689e-01, -4.809059259477282e-01, -4.828298594936981e-01, -5.034379036945840e-01, -5.034379036945840e-01, -1.431489574798253e-01, -1.453736057102468e-01, -6.389802231387000e-01, -1.064982743239212e-01, -1.438319089357821e-01, -1.438319089357821e-01, -9.798318504449418e-03, -1.101094372011419e-02, -7.293187584905318e-02, -1.180325769272544e-03, -1.088096538559627e-02, -1.088096538559627e-02, -3.556121404128410e+00, -3.557419538423813e+00, -3.556253667724608e+00, -3.557263081567353e+00, -3.556787519815333e+00, -3.556787519815333e+00, -1.310722316971599e+00, -1.320111829976724e+00, -1.305449930554980e+00, -1.312747435830919e+00, -1.323992116118409e+00, -1.323992116118409e+00, -4.351190965821178e-01, -4.737183867334135e-01, -4.178949833738865e-01, -4.374678340602874e-01, -4.501460338784405e-01, -4.501460338784405e-01, -8.630989785101459e-02, -1.414482776225243e-01, -8.428537078493481e-02, -1.369264295827695e+00, -9.236328081784827e-02, -9.236328081784827e-02, -1.096763774806953e-03, -1.586630229758332e-03, -1.137247145143196e-03, -7.208546969820465e-02, -1.457232570100359e-03, -1.457232570100359e-03, -4.503095571841951e-01, -4.472526380536696e-01, -4.484538341729605e-01, -4.492766351135608e-01, -4.488778229112392e-01, -4.488778229112392e-01, -4.364786381056822e-01, -3.708685928737466e-01, -3.905506268993057e-01, -4.090677180914722e-01, -3.997548509422384e-01, -3.997548509422384e-01, -4.961988364957328e-01, -1.752082647733225e-01, -2.083032376951744e-01, -2.711378783535810e-01, -2.369711045572275e-01, -2.369711045572274e-01, -3.402848437835850e-01, -7.442911134576169e-02, -7.564932468008358e-02, -2.611841792195604e-01, -7.409810025843203e-02, -7.409810025843196e-02, -1.306617751656960e-02, -5.619762720290724e-05, -3.095076505577164e-04, -7.064698449011317e-02, -1.099161245754988e-03, -1.099161245754983e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwb1k_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.937248813966253e-09, -3.937232042614981e-09, -3.937101547923669e-09, -3.937377521888099e-09, -3.937240431909670e-09, -3.937240431909670e-09, -4.201361139388881e-06, -4.201505874071903e-06, -4.205360335358194e-06, -4.203531459176324e-06, -4.201319248578602e-06, -4.201319248578602e-06, -1.246292889046641e-03, -1.267236618469807e-03, -1.764161735086636e-03, -1.770147561674832e-03, -1.254004998778179e-03, -1.254004998778179e-03, -2.325478978902112e-01, -2.214987824665658e-01, 3.971022012633095e-04, -5.728783169271374e-01, -2.291264700450523e-01, -2.291264700450523e-01, 2.725294607007708e+02, 2.658222941810251e+02, 5.186825841130737e+00, 2.553537117307191e+02, 2.763070579474461e+02, 2.763070579474461e+02, -1.089829659132862e-06, -1.090317432041963e-06, -1.089879556961888e-06, -1.090258590121204e-06, -1.090079760240542e-06, -1.090079760240542e-06, -3.489374365664683e-05, -3.424340909986686e-05, -3.499370400777385e-05, -3.448530981849306e-05, -3.431868372210678e-05, -3.431868372210678e-05, -2.573769901815375e-03, 2.011553791492178e-04, -2.261479173039370e-03, 1.572710462728655e-03, -2.368958334903339e-03, -2.368958334903339e-03, -1.119279436362473e+00, -1.986995378707538e-01, -1.241476851209356e+00, -2.909801103075016e-05, -9.160433374907645e-01, -9.160433374907645e-01, 2.721897196772896e+02, 2.640495856423066e+02, 7.604844024862942e+02, -1.800665688496174e+00, 3.896585405980823e+02, 3.896585405980817e+02, -2.566450255399238e-03, -2.403516971948664e-03, -2.306208969811996e-03, -2.330960229451633e-03, -2.301549149416642e-03, -2.301549149416639e-03, -1.968464538560156e-03, -6.436152493284256e-03, -5.217415597850205e-03, -3.907994773218963e-03, -4.575679827767396e-03, -4.575679827767391e-03, 4.398191552387855e-04, -9.487119412693520e-02, -5.319368386354009e-02, -1.737334645356059e-02, -3.340838556698879e-02, -3.340838556698879e-02, -8.137413926985913e-03, 9.529013456919669e+00, -1.472442880812297e-01, -1.275812753494970e-02, -2.062293162206649e+00, -2.062293162206652e+00, 1.942289994898246e+02, 5.953565126712812e+02, 4.450288254093556e+02, -2.509434333448298e+00, 5.787280506901292e+02, 5.787280506901278e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwb1k_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwb1k_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.453792881020835e-05, -1.453815844837191e-05, -1.453871124683689e-05, -1.453496057061442e-05, -1.453805385589584e-05, -1.453805385589584e-05, -2.775436576906803e-04, -2.775855508915998e-04, -2.787195116710247e-04, -2.766666961361983e-04, -2.775631483849621e-04, -2.775631483849621e-04, -2.355930769489709e-03, -2.326155936325914e-03, -1.624213943089691e-03, -1.639251645310494e-03, -2.345189621909301e-03, -2.345189621909301e-03, -3.960154996545132e-03, -4.321891409453982e-03, -2.589762119739679e-03, -1.210365121099308e-03, -4.071359626015101e-03, -4.071359626015101e-03, -2.971904688411089e-06, -3.390279689868400e-06, -6.932569228204634e-05, -2.686526026535870e-07, -3.421491950835473e-06, -3.421491950835474e-06, -2.270364236199089e-04, -2.280904403504870e-04, -2.271411325631901e-04, -2.279606871468378e-04, -2.275802814273013e-04, -2.275802814273013e-04, -2.619657172859747e-04, -2.670105022861876e-04, -2.519477275473951e-04, -2.557893556152959e-04, -2.791220374975574e-04, -2.791220374975574e-04, -6.561847842347336e-03, -1.240282339943801e-02, -6.733639062756232e-03, -1.215298748333417e-02, -6.923214536847441e-03, -6.923214536847441e-03, -4.299701434995194e-04, -1.456751740288871e-03, -4.869407601727779e-04, -1.934623506917164e-03, -8.147950457462470e-04, -8.147950457462470e-04, -2.575035527888732e-07, -3.760377772443233e-07, -4.770189010703516e-07, -1.807137943766328e-04, -4.274533259165149e-07, -4.274533259165147e-07, -1.720174471106644e-02, -1.414907958778020e-02, -1.514525674058078e-02, -1.598746071056006e-02, -1.555812503198624e-02, -1.555812503198624e-02, -1.879262873722554e-02, -4.786032157844813e-03, -6.887839697489260e-03, -9.933193977300818e-03, -8.256756186165932e-03, -8.256756186165934e-03, -1.114619740287045e-02, -2.039219582609204e-03, -3.148116437615883e-03, -6.753927378774448e-03, -4.714931968855889e-03, -4.714931968855887e-03, -5.187298369985341e-03, -4.672243692400858e-05, -1.115648744901561e-04, -1.094343744547512e-02, -4.028470824938104e-04, -4.028470824938113e-04, -3.464829181194732e-06, -1.603449443640014e-08, -8.721126006235346e-08, -4.396724529975842e-04, -3.942186910684218e-07, -3.942186910684204e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05