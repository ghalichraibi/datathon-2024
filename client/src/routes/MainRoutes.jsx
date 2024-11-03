import { lazy } from 'react';

// project import
import Loadable from 'components/Loadable';
import Dashboard from 'layout/Dashboard';
import HeroPage from 'pages/hero';
import UploadPage from 'pages/upload';

const Color = Loadable(lazy(() => import('pages/component-overview/color')));
const Typography = Loadable(lazy(() => import('pages/component-overview/typography')));
const Shadow = Loadable(lazy(() => import('pages/component-overview/shadows')));
const DashboardDefault = Loadable(lazy(() => import('pages/dashboard/index')));

// render - sample page
const SamplePage = Loadable(lazy(() => import('pages/extra-pages/sample-page')));

// ==============================|| MAIN ROUTING ||============================== //

const MainRoutes = {
  path: '/',
  children: [
    {
      path: '',
      element: <HeroPage />
    },
    {
      path: 'color',
      element: <Color />
    },
    {
      path: 'upload',
      element: <Dashboard />,
      children: [
        {
          path: '',
          element: <UploadPage />
        }
      ]
    },
    {
      path: 'dashboard',
      element: <Dashboard />,
      children: [
        {
          path: '',
          element: <DashboardDefault />
        },
        {
          path
        }
      ]
    },
    {
      path: 'sample-page',
      element: <SamplePage />
    },
    {
      path: 'shadow',
      element: <Shadow />
    },
    {
      path: 'typography',
      element: <Typography />
    }
  ]
};

export default MainRoutes;
