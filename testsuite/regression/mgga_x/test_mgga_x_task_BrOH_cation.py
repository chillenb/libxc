
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_task_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.300539051921790e+01, -2.300546641955271e+01, -2.300592088660147e+01, -2.300479563643888e+01, -2.300536855809676e+01, -2.300536855809676e+01, -3.309343258541259e+00, -3.309544076595879e+00, -3.315153484749774e+00, -3.315868938305560e+00, -3.314318434377453e+00, -3.314318434377453e+00, -4.817600308798814e-01, -4.803305729317595e-01, -4.488472045380313e-01, -4.736404558047321e-01, -4.751075889441193e-01, -4.751075889441193e-01, -1.746833471070867e-01, -1.762417000668209e-01, -4.185873928026887e-01, -8.487088934642148e-02, -1.546825737241357e-01, -1.546825737241356e-01, -3.273200755191148e-03, -3.524214538690914e-03, -2.631941443975314e-02, -1.532237520151762e-03, -2.475854925312282e-03, -2.475854925312282e-03, -5.664395646435854e+00, -5.665547334652668e+00, -5.664545763054697e+00, -5.665558226934113e+00, -5.664929644899303e+00, -5.664929644899303e+00, -2.034452337691137e+00, -2.056947501228369e+00, -2.027215551579168e+00, -2.047726626104266e+00, -2.050590193040923e+00, -2.050590193040923e+00, -6.087590375767529e-01, -6.653012034833368e-01, -5.362948270004451e-01, -5.582731744630287e-01, -6.215612335733001e-01, -6.215612335733002e-01, -5.583121053910102e-02, -1.666033538090163e-01, -5.177361513200589e-02, -1.922369173937616e+00, -7.976755862380955e-02, -7.976755862380960e-02, -1.099306836071413e-03, -1.507992542512332e-03, -1.241652957307626e-03, -3.836273467660855e-02, -1.585809445612707e-03, -1.585809445612707e-03, -6.348193793001782e-01, -6.311422326989855e-01, -6.324440664318595e-01, -6.335089217705649e-01, -6.329735952397548e-01, -6.329735952397548e-01, -6.129663049034026e-01, -5.251900026707561e-01, -5.501426726738603e-01, -5.751049225888611e-01, -5.619508143808836e-01, -5.619508143808837e-01, -6.905566207345860e-01, -2.228190379260751e-01, -2.650040998189024e-01, -3.425633681539035e-01, -3.045397464356155e-01, -3.045397464356155e-01, -4.688950294400758e-01, -2.538793985051891e-02, -3.324758260070997e-02, -3.390958859595302e-01, -4.662470506058534e-02, -4.662470506058535e-02, -5.071720003202738e-03, -2.785281515255958e-04, -7.371092485590952e-04, -4.892425308814216e-02, -1.436778652956436e-03, -1.436778652956435e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_task_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.824726042971112e+01, -2.824341116633461e+01, -2.824739366051401e+01, -2.824350843774967e+01, -2.824723683286388e+01, -2.824344653267950e+01, -2.824570962797484e+01, -2.824150433590689e+01, -2.824735470368631e+01, -2.824185363256211e+01, -2.824735470368631e+01, -2.824185363256211e+01, -5.063442799468340e+00, -5.064460905241732e+00, -5.063349426074419e+00, -5.064493201038374e+00, -5.062631078839391e+00, -5.062832335895575e+00, -5.058722579494681e+00, -5.059794308365588e+00, -5.062734807572538e+00, -5.059154788397710e+00, -5.062734807572538e+00, -5.059154788397710e+00, -9.220910138156418e-01, -9.326089346915251e-01, -9.185316319387927e-01, -9.310473332401205e-01, -8.659975109772324e-01, -8.497140020992275e-01, -8.733142594501999e-01, -8.807641678431677e-01, -9.445121560520584e-01, -7.567353744022489e-01, -9.445121560520584e-01, -7.567353744022489e-01, -1.056202092386465e-01, -1.721607005363756e-01, -1.060054100309668e-01, -1.876997345883751e-01, -8.374407820733240e-01, -9.310849042776556e-01, -3.153904743265729e-02, 2.115182524024201e-02, -1.005616628686553e-01, -1.940930160253763e-02, -1.005616628686554e-01, -1.940930160253755e-02, -5.600258524179869e-03, -5.994450982438487e-03, -5.969849170787926e-03, -6.499121759987226e-03, -3.141647426242005e-02, -3.122667157788160e-02, -2.854765272940076e-03, -2.784723753003404e-03, -4.837710762951321e-03, -2.480717796180298e-03, -4.837710762951320e-03, -2.480717796180299e-03, -6.869205024645902e+00, -6.866164319364597e+00, -6.876112746684761e+00, -6.872848993852367e+00, -6.869108179031196e+00, -6.866046932919274e+00, -6.875056857042606e+00, -6.872139669743701e+00, -6.873107646391480e+00, -6.869583973309823e+00, -6.873107646391480e+00, -6.869583973309823e+00, -2.548346602175334e+00, -2.548702719535568e+00, -2.569165972279911e+00, -2.565308574213051e+00, -2.509975549392022e+00, -2.520983555947990e+00, -2.509033653904781e+00, -2.521440078716790e+00, -2.589040279862359e+00, -2.564145591819283e+00, -2.589040279862359e+00, -2.564145591819283e+00, -8.231073337039532e-01, -8.204285013484146e-01, -9.145372148763458e-01, -9.138004040474501e-01, -7.685191809511949e-01, -7.846200074501231e-01, -8.397415364728081e-01, -8.532750070151082e-01, -8.573892761117468e-01, -8.169603888614850e-01, -8.573892761117469e-01, -8.169603888614850e-01, 2.142436494868916e-03, -2.742554285764121e-03, -1.474488951702319e-02, -9.938890557721441e-03, -1.374578633857183e-02, -1.683129784242849e-03, -2.874088481340590e+00, -2.872709188145047e+00, 5.160207729905097e-02, 8.245069409488801e-02, 5.160207729905141e-02, 8.245069409488862e-02, -1.996443220683844e-03, -2.076544168127086e-03, -2.755015830194206e-03, -2.792053979315796e-03, -2.161022352778617e-03, -2.351098925470998e-03, -2.646199730787558e-02, -1.993746857316527e-02, -2.097546836465126e-03, -3.187868038980580e-03, -2.097546836465126e-03, -3.187868038980581e-03, -8.471531987988257e-01, -8.506213256393180e-01, -8.288039480624625e-01, -8.322600305370323e-01, -8.352865563483581e-01, -8.387829379942520e-01, -8.406577359540208e-01, -8.440825860622841e-01, -8.379795762662263e-01, -8.414371734854830e-01, -8.379795762662263e-01, -8.414371734854830e-01, -8.262986031024057e-01, -8.292465435242143e-01, -5.961845611142849e-01, -5.992789807179568e-01, -6.776221651199837e-01, -6.810841833009404e-01, -7.437087282981160e-01, -7.464739954658876e-01, -7.143586940468465e-01, -7.161407240626484e-01, -7.143586940468465e-01, -7.161407240626487e-01, -9.690043148110241e-01, -9.669344559692112e-01, -2.028136254047207e-01, -2.043674758548755e-01, -2.600046496088214e-01, -2.697068532226227e-01, -4.550913348267431e-01, -4.588326751452944e-01, -3.557652344170207e-01, -3.565819996898074e-01, -3.557652344170207e-01, -3.565819996898071e-01, -6.058012391783197e-01, -6.130812004559260e-01, -2.656209345236308e-02, -2.879160784042952e-02, -3.247891652320935e-02, -3.151993407398047e-02, -4.580679501074744e-01, -4.617899268195916e-01, -3.863079325737918e-03, 1.034533108808528e-02, -3.863079325738036e-03, 1.034533108808516e-02, -8.661708342502015e-03, -9.058544884224638e-03, -5.122215349584408e-04, -5.437925874928754e-04, -1.307312354735461e-03, -1.430174023619184e-03, 5.358943620433114e-03, 1.094936758208962e-01, -2.072403098861242e-03, -2.851762035317696e-03, -2.072403098861240e-03, -2.851762035317694e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.855946860477755e-08, 0.000000000000000e+00, -3.861288469353818e-08, -3.855972329296644e-08, 0.000000000000000e+00, -3.861306270961466e-08, -3.856702482078202e-08, 0.000000000000000e+00, -3.862171211896551e-08, -3.856450477321617e-08, 0.000000000000000e+00, -3.861841580733956e-08, -3.855949676968372e-08, 0.000000000000000e+00, -3.862447370855085e-08, -3.855949676968372e-08, 0.000000000000000e+00, -3.862447370855085e-08, -3.628311655726287e-05, 0.000000000000000e+00, -3.628326773128027e-05, -3.628123906108611e-05, 0.000000000000000e+00, -3.628190418829383e-05, -3.624045344420089e-05, 0.000000000000000e+00, -3.623067510731150e-05, -3.625411273091671e-05, 0.000000000000000e+00, -3.624986146323090e-05, -3.631083909734785e-05, 0.000000000000000e+00, -3.620706815338165e-05, -3.631083909734785e-05, 0.000000000000000e+00, -3.620706815338165e-05, -1.432856020193020e-02, 0.000000000000000e+00, -1.421074851525264e-02, -1.443886012130940e-02, 0.000000000000000e+00, -1.420196168029654e-02, -1.554543522013792e-02, 0.000000000000000e+00, -1.695549916924814e-02, -1.840708695848652e-02, 0.000000000000000e+00, -1.854149528103473e-02, -1.306976410649718e-02, 0.000000000000000e+00, -2.615969657075481e-02, -1.306976410649718e-02, 0.000000000000000e+00, -2.615969657075481e-02, -6.125050793389847e+00, 0.000000000000000e+00, -3.085188766452739e+00, -6.387923312328270e+00, 0.000000000000000e+00, -2.628523736824740e+00, 1.466467588198608e-03, 0.000000000000000e+00, 5.126582998572546e-04, -7.319016916831020e+00, 0.000000000000000e+00, -1.405257364450874e+01, -5.785325315138474e+00, 0.000000000000000e+00, -4.885019792754829e+00, -5.785325315138472e+00, 0.000000000000000e+00, -4.885019792754814e+00, 4.618589919456649e+02, 0.000000000000000e+00, 3.698109473479222e+02, 4.367380368780408e+02, 0.000000000000000e+00, 3.729293232537433e+02, -6.177726047180141e+00, 0.000000000000000e+00, -9.856122513626790e+00, 1.127868454952796e+03, 0.000000000000000e+00, 1.144188971365600e+03, 5.779346541245905e+02, 0.000000000000000e+00, 2.680161068176904e+03, 5.779346541245910e+02, 0.000000000000000e+00, 2.680161068176905e+03, -1.516367392835783e-05, 0.000000000000000e+00, -1.520122712249005e-05, -1.515250753625175e-05, 0.000000000000000e+00, -1.519037875076999e-05, -1.517212180796199e-05, 0.000000000000000e+00, -1.520740626184679e-05, -1.516217580980014e-05, 0.000000000000000e+00, -1.519719906710259e-05, -1.515109968843982e-05, 0.000000000000000e+00, -1.519463228598067e-05, -1.515109968843982e-05, 0.000000000000000e+00, -1.519463228598067e-05, -1.728126055824731e-04, 0.000000000000000e+00, -1.718249733019028e-04, -1.685765498861120e-04, 0.000000000000000e+00, -1.706148006353846e-04, -1.781871907460960e-04, 0.000000000000000e+00, -1.762379476575456e-04, -1.881334717185960e-04, 0.000000000000000e+00, -1.853589937943961e-04, -1.653169081018983e-04, 0.000000000000000e+00, -1.683845227047066e-04, -1.653169081018983e-04, 0.000000000000000e+00, -1.683845227047066e-04, -2.508696334898342e-02, 0.000000000000000e+00, -2.420928912803567e-02, -1.654465073849365e-02, 0.000000000000000e+00, -1.584885445630495e-02, -6.515426782537867e-02, 0.000000000000000e+00, -4.896754487257118e-02, -5.235510556028524e-02, 0.000000000000000e+00, -4.281926460824471e-02, -1.702656565583865e-02, 0.000000000000000e+00, -2.724332852945437e-02, -1.702656565583867e-02, 0.000000000000000e+00, -2.724332852945432e-02, -1.362101415710386e+01, 0.000000000000000e+00, -1.215137615250799e+01, -5.935863980749788e+00, 0.000000000000000e+00, -6.020192633018754e+00, -1.072130156006604e+01, 0.000000000000000e+00, -1.419529941017977e+01, -3.395646060554654e-04, 0.000000000000000e+00, -3.400701613810766e-04, -2.359603476739262e+01, 0.000000000000000e+00, -3.260097820605892e+01, -2.359603476739270e+01, 0.000000000000000e+00, -3.260097820605903e+01, 2.259467491232499e+03, 0.000000000000000e+00, 1.947530461412869e+03, 1.342165384964042e+03, 0.000000000000000e+00, 1.250072199799083e+03, 5.909260078407697e+03, 0.000000000000000e+00, 5.647903282037073e+03, -1.148859119120719e+01, 0.000000000000000e+00, -2.042129698510699e+01, 3.864487043650032e+03, 0.000000000000000e+00, 2.160606906471026e+03, 3.864487043650025e+03, 0.000000000000000e+00, 2.160606906471025e+03, -5.856898395325106e-02, 0.000000000000000e+00, -5.838691872588991e-02, -6.080732372546642e-02, 0.000000000000000e+00, -6.056989758229596e-02, -5.998451567314567e-02, 0.000000000000000e+00, -5.977494524775275e-02, -5.934065466162344e-02, 0.000000000000000e+00, -5.912809074767936e-02, -5.964311093140404e-02, 0.000000000000000e+00, -5.943827891566170e-02, -5.964311093140404e-02, 0.000000000000000e+00, -5.943827891566169e-02, -4.249434472218122e-02, 0.000000000000000e+00, -4.315382178877135e-02, -1.105768574183493e-01, 0.000000000000000e+00, -1.097232149728919e-01, -7.901793261451096e-02, 0.000000000000000e+00, -7.862868693234855e-02, -5.937138640048814e-02, 0.000000000000000e+00, -5.887528588488162e-02, -6.469582718971910e-02, 0.000000000000000e+00, -6.567834909218735e-02, -6.469582718971907e-02, 0.000000000000000e+00, -6.567834909218730e-02, -1.754311440797184e-02, 0.000000000000000e+00, -1.623637907950800e-02, -1.257606987035745e+00, 0.000000000000000e+00, -1.244180821638300e+00, -9.001135208339279e-01, 0.000000000000000e+00, -8.413161162252685e-01, -3.617164197511579e-01, 0.000000000000000e+00, -3.560102900075494e-01, -5.317178907585786e-01, 0.000000000000000e+00, -5.317328241928966e-01, -5.317178907585787e-01, 0.000000000000000e+00, -5.317328241928970e-01, -7.455407208306748e-02, 0.000000000000000e+00, -6.532254524232162e-02, -2.640310861125438e+01, 0.000000000000000e+00, -1.764237641520314e+01, -9.699242074473471e+00, 0.000000000000000e+00, -1.111191692393455e+01, -4.048165356693732e-01, 0.000000000000000e+00, -2.625403834821947e-01, -2.436861457651036e+01, 0.000000000000000e+00, -3.370445978684548e+01, -2.436861457651026e+01, 0.000000000000000e+00, -3.370445978684549e+01, 2.055469055206936e+02, 0.000000000000000e+00, 1.930040638814120e+02, 2.623898106133392e+04, 0.000000000000000e+00, 3.649503190021660e+04, 5.992222246643483e+03, 0.000000000000000e+00, 5.584863304030678e+03, -3.544434712348589e+01, 0.000000000000000e+00, -1.031639123121905e+02, 6.358050852747939e+03, 0.000000000000000e+00, 2.585429786591940e+03, 6.358050852747955e+03, 0.000000000000000e+00, 2.585429786591949e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.041999197839604e-03, 3.046072550495765e-03, 3.042046242412526e-03, 3.046106219915868e-03, 3.042699336160940e-03, 3.046901767343629e-03, 3.042195075787443e-03, 3.046252077675379e-03, 3.042017464410720e-03, 3.046879048172933e-03, 3.042017464410720e-03, 3.046879048172933e-03, 1.341960990170448e-02, 1.342130264633591e-02, 1.341856941694785e-02, 1.342071304755202e-02, 1.339868380987624e-02, 1.339513525432648e-02, 1.340075335949290e-02, 1.340149146440576e-02, 1.342399298575594e-02, 1.338214064855618e-02, 1.342399298575594e-02, 1.338214064855618e-02, 4.611687558903664e-02, 4.648128197208546e-02, 4.611645796718192e-02, 4.637956017647103e-02, 4.507748141230746e-02, 4.661137887507151e-02, 5.083349816330052e-02, 5.152262854992775e-02, 4.507120915004718e-02, 5.861588443357713e-02, 4.507120915004718e-02, 5.861588443357713e-02, 2.429400513472187e-01, 1.402608687619682e-01, 2.625505446400846e-01, 1.266635553806967e-01, 1.042962745062363e-02, 1.414724145264030e-02, 1.154982078042702e-01, 2.514602174758707e-01, 2.605507463703748e-01, 1.268403859956275e-02, 2.605507463703747e-01, 1.268403859956275e-02, 8.271030060809608e-06, 8.388888981810466e-05, 4.144483310081937e-06, 2.344221620728357e-05, 1.192506934082973e-03, 2.450053140465391e-03, 1.605684690503482e-09, 1.926179971278002e-09, 3.494070110327619e-07, 9.268765216393927e-10, 3.494070110327700e-07, 9.268765216357512e-10, 1.715333116170636e-02, 1.718295257516249e-02, 1.714947286191056e-02, 1.717918287208199e-02, 1.716321687256208e-02, 1.719011480056871e-02, 1.715950651983408e-02, 1.718630359058135e-02, 1.714371959597131e-02, 1.717978489369459e-02, 1.714371959597131e-02, 1.717978489369459e-02, 1.119983265811916e-02, 1.113471039245758e-02, 1.114872289863075e-02, 1.126040421403743e-02, 1.128943510845292e-02, 1.123520494595029e-02, 1.206351626003956e-02, 1.196315760397110e-02, 1.102591148304605e-02, 1.105428590185462e-02, 1.102591148304605e-02, 1.105428590185462e-02, 4.447953398729288e-02, 4.268368812739329e-02, 3.674244395292792e-02, 3.526370639965776e-02, 8.605239340345108e-02, 7.138963668857717e-02, 7.798722532200524e-02, 6.957056060361409e-02, 3.434693985077728e-02, 4.652296780283447e-02, 3.434693985077730e-02, 4.652296780283437e-02, 6.760616916412594e-02, 6.019089521865971e-02, 2.466862917862231e-01, 2.537272547763531e-01, 3.319020797834098e-02, 5.953352865433147e-02, 2.059327521830041e-02, 2.059515583392571e-02, 1.996670054223685e-01, 3.398460448523047e-01, 1.996670054223692e-01, 3.398460448523059e-01, 2.291952005982361e-11, 2.673260289112358e-11, 4.016478814423178e-10, 3.100618635379018e-10, 6.881879112615085e-09, 1.115415499180188e-08, 6.994356488639050e-03, 2.143319489648071e-02, 5.259336015588779e-11, 4.620253129329883e-07, 5.259336015602260e-11, 4.620253129331612e-07, 9.546720811273826e-02, 9.624627807433944e-02, 9.796226121336299e-02, 9.868321774762114e-02, 9.705301975855428e-02, 9.780808442258450e-02, 9.633637768018018e-02, 9.707334518408815e-02, 9.666603889427924e-02, 9.742152850249391e-02, 9.666603889427924e-02, 9.742152850249389e-02, 6.432043942489067e-02, 6.591700084435781e-02, 1.073173557873245e-01, 1.078411855858247e-01, 8.855523983029587e-02, 8.923787322918747e-02, 7.599287576826266e-02, 7.612714205673597e-02, 7.773941853374447e-02, 7.967760385882433e-02, 7.773941853374440e-02, 7.967760385882418e-02, 4.465150335541491e-02, 4.154445175724026e-02, 1.082850090243607e-01, 1.089820255397877e-01, 1.330505284868770e-01, 1.284013825881276e-01, 1.279132004320219e-01, 1.280733016861378e-01, 1.227433115010188e-01, 1.226605320287046e-01, 1.227433115010188e-01, 1.226605320287046e-01, 5.946894858344131e-02, 5.353682952620767e-02, 7.613994516359983e-03, 5.012482194251883e-03, 3.530337146198613e-03, 4.838103632100097e-03, 1.278385200960198e-01, 8.747508130878860e-02, 5.438368992360930e-02, 9.526677827330214e-02, 5.438368992360902e-02, 9.526677827330211e-02, 7.217671367338974e-08, 8.067862849392453e-08, 2.455683922580734e-14, 3.156791235590030e-14, 5.829088519619531e-10, 9.386498431772563e-10, 7.402495279906762e-02, 2.568567789221686e-01, 2.521988715156364e-10, 2.607923110879038e-07, 2.521988715167904e-10, 2.607923110877925e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05