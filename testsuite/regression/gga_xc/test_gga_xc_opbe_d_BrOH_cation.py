
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_opbe_d_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opbe_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.099649490674530e+01, -2.099652267119993e+01, -2.099671324982832e+01, -2.099629879475309e+01, -2.099650641283907e+01, -2.099650641283907e+01, -3.524006478419620e+00, -3.523972085252516e+00, -3.523262635960604e+00, -3.525156769393920e+00, -3.524049286882188e+00, -3.524049286882188e+00, -7.331812882637889e-01, -7.330328427107079e-01, -7.316191673907656e-01, -7.365407667969669e-01, -7.373872696630872e-01, -7.373872696630872e-01, -2.326060440945097e-01, -2.338245409278534e-01, -8.437024477529933e-01, -1.999643805076548e-01, -2.237889605194943e-01, -2.237889605194943e-01, -1.232312006811251e-02, -1.297517514177511e-02, -6.951190837981919e-02, -7.110625169361616e-03, -9.920866366989002e-03, -9.920866366989002e-03, -5.099720661380994e+00, -5.099302723040186e+00, -5.099711219519762e+00, -5.099342072991937e+00, -5.099503165475465e+00, -5.099503165475465e+00, -2.154957149634615e+00, -2.164717222867226e+00, -2.156733024145180e+00, -2.165303489053690e+00, -2.159570164164410e+00, -2.159570164164410e+00, -6.249036177170425e-01, -6.611649083614780e-01, -5.836549643858947e-01, -5.916636303291243e-01, -6.328353397752818e-01, -6.328353397752818e-01, -1.588178952117362e-01, -2.505604481335654e-01, -1.493578250218013e-01, -1.889835316078764e+00, -1.730840114269306e-01, -1.730840114269306e-01, -5.490243864599426e-03, -6.950976168555717e-03, -5.322182766638720e-03, -1.069626931674594e-01, -6.685111067325778e-03, -6.685111067325784e-03, -6.121389519606887e-01, -6.110972013226642e-01, -6.114193214664014e-01, -6.117167661915227e-01, -6.115632648527671e-01, -6.115632648527671e-01, -5.960600762222131e-01, -5.448618074388993e-01, -5.573489720258712e-01, -5.710598632752000e-01, -5.637350088125976e-01, -5.637350088125976e-01, -6.916136327932023e-01, -2.955120907594265e-01, -3.315816371022741e-01, -3.959267025546019e-01, -3.602545492577596e-01, -3.602545492577595e-01, -5.046229379528916e-01, -6.668950887987160e-02, -8.899287137299901e-02, -3.759048981886232e-01, -1.290549425075082e-01, -1.290549425075082e-01, -1.735500610103549e-02, -1.858432764519804e-03, -3.907805735229189e-03, -1.229240826858808e-01, -6.139477861387192e-03, -6.139477861387182e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_opbe_d_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opbe_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.522380476622673e+01, -2.522377578891103e+01, -2.522392623000956e+01, -2.522386435955500e+01, -2.522427808765658e+01, -2.522437699614188e+01, -2.522290269541773e+01, -2.522262892453630e+01, -2.522387503060397e+01, -2.522330305945882e+01, -2.522387503060397e+01, -2.522330305945882e+01, -4.135150514535665e+00, -4.135395483993215e+00, -4.135205025429463e+00, -4.135461434648257e+00, -4.136640699122714e+00, -4.137008713226269e+00, -4.134805598393679e+00, -4.135188127868299e+00, -4.134391220230009e+00, -4.136492760733835e+00, -4.134391220230009e+00, -4.136492760733835e+00, -8.308170207067734e-01, -8.363482985554493e-01, -8.285280474067802e-01, -8.352921688482825e-01, -7.949134631676855e-01, -7.863881737481395e-01, -7.962894194308708e-01, -7.990672372473997e-01, -8.389769205708264e-01, -7.565936925261033e-01, -8.389769205708264e-01, -7.565936925261033e-01, -2.198722596485285e-01, -2.238302891988391e-01, -2.230227191302546e-01, -2.279521341475852e-01, -9.493561754249283e-01, -9.887468418287676e-01, -1.806069599946279e-01, -1.817227270831660e-01, -2.051552565998057e-01, -1.766936522787434e-01, -2.051552565998057e-01, -1.766936522787432e-01, -1.585475174926287e-02, -1.684072220255894e-02, -1.660698415340697e-02, -1.778075323062524e-02, -8.530920923998182e-02, -8.919864258842211e-02, -9.551159912962076e-03, -9.392502549593338e-03, -1.415340219987725e-02, -8.064362198991155e-03, -1.415340219987726e-02, -8.064362198991155e-03, -6.331773351725400e+00, -6.330252763318667e+00, -6.334878407799249e+00, -6.333258452631243e+00, -6.331941758577511e+00, -6.330356079557462e+00, -6.334610301416622e+00, -6.333082329201646e+00, -6.333364880445271e+00, -6.331763874785016e+00, -6.333364880445271e+00, -6.331763874785016e+00, -2.213438906238264e+00, -2.213325841442680e+00, -2.233605051949201e+00, -2.232956500512093e+00, -2.189159260984919e+00, -2.195344404648204e+00, -2.206495409860141e+00, -2.212863897790847e+00, -2.242472520428864e+00, -2.227116942725195e+00, -2.242472520428864e+00, -2.227116942725195e+00, -7.688816774832040e-01, -7.674578727885243e-01, -8.556141777014499e-01, -8.563084705513423e-01, -6.992830840720549e-01, -7.214170742553020e-01, -7.517566413678978e-01, -7.696673868101349e-01, -7.987612829744449e-01, -7.672134785644483e-01, -7.987612829744448e-01, -7.672134785644482e-01, -1.605576825943125e-01, -1.602802768880850e-01, -2.242647802267373e-01, -2.247050188998032e-01, -1.526827820411861e-01, -1.553127242749999e-01, -2.452036302593994e+00, -2.451123477933354e+00, -1.657509634124988e-01, -1.587265818843190e-01, -1.657509634124988e-01, -1.587265818843190e-01, -7.167887219791970e-03, -7.449144325485118e-03, -9.189712933702951e-03, -9.328905248822095e-03, -6.869173184224422e-03, -7.272646082461310e-03, -1.218878311408245e-01, -1.228045310536545e-01, -7.018675108570635e-03, -9.629392730216147e-03, -7.018675108570642e-03, -9.629392730216152e-03, -7.985348891734450e-01, -8.009786603310021e-01, -7.902888802175300e-01, -7.928234287090584e-01, -7.933017798492187e-01, -7.958302339155263e-01, -7.957285839741857e-01, -7.981816302243776e-01, -7.945269809012176e-01, -7.970165816681944e-01, -7.945269809012176e-01, -7.970165816681944e-01, -7.792567227635147e-01, -7.811936377697511e-01, -6.283759718199371e-01, -6.309036380006312e-01, -6.734891104855200e-01, -6.763180021326469e-01, -7.184509583519756e-01, -7.205211124688389e-01, -6.960289365635961e-01, -6.981640636593860e-01, -6.960289365635961e-01, -6.981640636593860e-01, -8.931141855902205e-01, -8.948705200178678e-01, -2.717458679516251e-01, -2.725435578457123e-01, -3.257504119682096e-01, -3.280345624743710e-01, -4.534690683437381e-01, -4.554708014179759e-01, -3.847078859957468e-01, -3.847832288018800e-01, -3.847078859957467e-01, -3.847832288018798e-01, -5.810037899369507e-01, -5.848096014156737e-01, -8.423949188881140e-02, -8.472187033211732e-02, -1.068734990173614e-01, -1.092609158479103e-01, -4.471639592216224e-01, -4.526364091531766e-01, -1.360544193470048e-01, -1.342677538582317e-01, -1.360544193470047e-01, -1.342677538582316e-01, -2.261851684673847e-02, -2.342664094238210e-02, -2.474833587869415e-03, -2.480357905804215e-03, -5.033922355351881e-03, -5.353522813871543e-03, -1.310883661674583e-01, -1.322318428435334e-01, -6.646003473132259e-03, -8.828741139340774e-03, -6.646003473132247e-03, -8.828741139340760e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_opbe_d_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opbe_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.412735248367669e-08, 3.923612199742372e-10, -1.412742993622268e-08, -1.412721021144043e-08, 3.923676753213836e-10, -1.412732566955726e-08, -1.412666329663915e-08, 3.923756277517287e-10, -1.412657117311175e-08, -1.412828388294346e-08, 3.922800328124702e-10, -1.412862285153992e-08, -1.412728858689751e-08, 3.923330647602070e-10, -1.412772310819044e-08, -1.412728858689751e-08, 3.923330647602070e-10, -1.412772310819044e-08, -1.859295758465213e-05, 2.161140870393403e-06, -1.859728059043493e-05, -1.859292013104289e-05, 2.161537533446051e-06, -1.859760217486825e-05, -1.859596990287201e-05, 2.170479464327282e-06, -1.859748133151356e-05, -1.857897788731460e-05, 2.152981753563923e-06, -1.858144888944161e-05, -1.860472906973760e-05, 2.161362979902375e-06, -1.858231701525968e-05, -1.860472906973760e-05, 2.161362979902375e-06, -1.858231701525968e-05, -1.133884540417338e-02, 3.506785440253675e-03, -1.127730918357445e-02, -1.137912780082530e-02, 3.488355604221912e-03, -1.130402929445663e-02, -1.202832426084021e-02, 2.998318647151344e-03, -1.211015132177196e-02, -1.172603966910941e-02, 2.923136417226994e-03, -1.170538401769293e-02, -1.136315424740304e-02, 2.962058419361950e-03, -1.188139221143326e-02, -1.136315424740304e-02, 2.962058419361950e-03, -1.188139221143326e-02, -1.541150709532486e+00, 4.019686817172466e-01, -1.401075425770022e+00, -1.520256184971859e+00, 4.152264525026472e-01, -1.355789444896269e+00, -6.622777567647928e-03, 1.880495689839574e-03, -5.962692040056360e-03, -2.518711552613779e+00, 3.598752703017317e-01, -2.423364178989140e+00, -1.349685040720923e+00, 2.872163228768307e-01, -4.225844065409202e+00, -1.349685040720922e+00, 2.872163228768315e-01, -4.225844065409206e+00, -1.130032862989840e+01, 1.753408940560727e-02, -1.118501476667081e+01, -1.190334356115527e+01, 2.062247647740581e-02, -1.183966504177729e+01, -6.322592476465013e+00, 1.042980837415839e-01, -6.329385932821117e+00, -1.036512141763375e+01, 6.230785417031861e-03, -1.008389291655463e+01, -1.128888514686119e+01, 9.140821999037072e-03, -2.866584068628350e+01, -1.128888514686146e+01, 9.140821998959816e-03, -2.866584068628381e+01, -3.887826446836686e-06, 5.453911403196199e-07, -3.891720339863822e-06, -3.885408902858202e-06, 5.486777504086519e-07, -3.889329338725860e-06, -3.887691174180772e-06, 5.455220463559742e-07, -3.891609980925487e-06, -3.885576664784609e-06, 5.484231827494280e-07, -3.889480479901582e-06, -3.886615296479323e-06, 5.470632732687274e-07, -3.890512847453158e-06, -3.886615296479323e-06, 5.470632732687274e-07, -3.890512847453158e-06, -1.576289209418765e-04, 1.377231419222379e-05, -1.576592050164776e-04, -1.539808680432392e-04, 1.367499371250295e-05, -1.541309170609370e-04, -1.590105040026702e-04, 1.324398916634196e-05, -1.587567574817475e-04, -1.558611779091629e-04, 1.316341313911574e-05, -1.555387747343226e-04, -1.545095746595456e-04, 1.398076098423155e-05, -1.555545320788206e-04, -1.545095746595456e-04, 1.398076098423155e-05, -1.555545320788206e-04, -1.948113410887868e-02, 1.020348451350403e-02, -1.975102079727573e-02, -1.361337787974031e-02, 1.132402530458093e-02, -1.359601677777112e-02, -2.882388904373638e-02, 1.343295495017910e-02, -2.439659040626558e-02, -2.436317991357438e-02, 1.759707959929150e-02, -2.026408126143578e-02, -1.628469247599969e-02, 1.005377879741530e-02, -2.100166792354321e-02, -1.628469247599969e-02, 1.005377879741530e-02, -2.100166792354321e-02, -3.540781943159644e+00, 2.333664680371389e-01, -3.579231053091750e+00, -1.072484472165672e+00, 1.709370172456237e-01, -1.059122849321996e+00, -4.120987198039013e+00, 2.397573581301675e-01, -3.925130035644971e+00, -1.844542312075715e-04, 9.986390346942655e-05, -1.848870126499290e-04, -3.493392120078054e+00, 4.127957276624746e-01, -3.940473920523347e+00, -3.493392120078054e+00, 4.127957276624746e-01, -3.940473920523347e+00, -1.455440370469121e+01, 6.075681598427547e-03, -1.259691822124685e+01, -1.252917006344095e+01, 7.660602868189185e-03, -1.156765424890349e+01, -7.133928533942267e+01, 7.630462532669158e-02, -7.935498832827216e+01, -6.678760481187177e+00, 2.459245620175788e-01, -6.435339386009199e+00, -3.548627922079023e+01, 2.920395070844230e-02, -3.484159173983313e+01, -3.548627922078224e+01, 2.920395071840575e-02, -3.484159173982523e+01, -1.804174798281425e-02, 1.805618894070567e-02, -1.763566938040615e-02, -1.905047483581404e-02, 1.625118725274891e-02, -1.864579849610257e-02, -1.871867080887310e-02, 1.684021865109997e-02, -1.831331394650378e-02, -1.842632828046501e-02, 1.736430753106517e-02, -1.802092176064865e-02, -1.857473659238707e-02, 1.709805764101786e-02, -1.816931887462860e-02, -1.857473659238707e-02, 1.709805764101786e-02, -1.816931887462860e-02, -1.984759920211523e-02, 2.101674511912313e-02, -1.945085479353837e-02, -3.805697888974732e-02, 1.505850693284694e-02, -3.739633051033355e-02, -3.259670970816448e-02, 1.620043832429425e-02, -3.198149414054924e-02, -2.742595475816230e-02, 1.772481063252601e-02, -2.694446883173892e-02, -3.003338362443199e-02, 1.694524489667253e-02, -2.949863658557680e-02, -3.003338362443199e-02, 1.694524489667253e-02, -2.949863658557680e-02, -1.139891725504262e-02, 9.085721683587256e-03, -1.131784842220344e-02, -5.642166798867766e-01, 1.148842933935789e-01, -5.558174644142527e-01, -3.433640741126385e-01, 9.313371835463204e-02, -3.346768045131647e-01, -1.448994485582713e-01, 6.583043239933262e-02, -1.414743866873297e-01, -2.262326517505636e-01, 8.207555251242558e-02, -2.274036595870283e-01, -2.262326517505638e-01, 8.207555251242564e-02, -2.274036595870284e-01, -5.256712228634170e-02, 2.156832701203833e-02, -5.117418799728252e-02, -5.805869469079093e+00, 8.590054229718191e-02, -5.809793276638759e+00, -5.535721048972543e+00, 1.299073903632207e-01, -5.627627038329766e+00, -1.754968679496114e-01, 9.683169819728893e-02, -1.640074663395570e-01, -6.164836985009523e+00, 4.111596468209503e-01, -6.989049517920265e+00, -6.164836985009529e+00, 4.111596468209449e-01, -6.989049517920268e+00, -8.940834061054717e+00, 2.184964381918247e-02, -9.125565609973187e+00, -4.479798548206548e+01, 8.309470079659752e-03, -7.935418711077004e+01, -2.764167981200287e+01, 1.039900992286666e-02, -2.942565909595408e+01, -6.954754930327250e+00, 3.819155966882047e-01, -6.779204906625545e+00, -7.316330897559757e+01, 3.727638362507801e-02, -3.617131802886736e+01, -7.316330897559654e+01, 3.727638364094096e-02, -3.617131802886634e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05